import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from vcd_core.vcd_add_noise import add_diffusion_noise_qwen2vl
from vcd_core.vcd_decoding import vcd_contrast, select_token

class Qwen2VLVCDWrapper:
    def __init__(self, model_path, device="cuda:0", dtype=torch.bfloat16):
        # Route target_device to GPU if receiving auto or cuda
        target_device = "cuda:0" if (device == "auto" or device == "cuda") else device
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Try native PyTorch SDPA (Scaled Dot-Product Attention) for acceleration on RTX 6000
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                device_map={"": target_device}
            ).eval()
        except Exception:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation="eager",
                device_map={"": target_device}
            ).eval()
        
        tok = self.processor.tokenizer
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        eos |= {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
        self.eos_ids = {e for e in eos if e is not None}

    def _build_inputs(self, images, prompt):
        is_batch = isinstance(images, (list, tuple))
        img_list = list(images) if is_batch else [images]
        prompts = [prompt] * len(img_list)

        messages_batch = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
            for p in prompts
        ]
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]

        # Use left padding for batched causal autoregressive generation
        if len(img_list) > 1:
            self.processor.tokenizer.padding_side = "left"
            inp = self.processor(
                text=texts,
                images=[im.convert("RGB") for im in img_list],
                padding=True,
                return_tensors="pt"
            )
        else:
            inp = self.processor(
                text=texts,
                images=[img_list[0].convert("RGB")],
                return_tensors="pt"
            )

        dev = self.model.device
        inp = {k: v.to(dev) for k, v in inp.items() if hasattr(v, "to")}
        return inp, is_batch, len(img_list)

    @torch.inference_mode()
    def generate(self, prompt, image, use_vcd=False, vcd_noise_step=500, vcd_alpha=1.0, vcd_beta=0.1, **gen_kwargs):
        """
        Supports both single image (image = PIL.Image) and batched images (image = [img1, img2, ...]).
        Returns str if image is a single image; returns list[str] if image is a list.
        """
        do_sample = gen_kwargs.get("do_sample", False)
        temperature = gen_kwargs.get("temperature", 0.7) if do_sample else 1.0
        top_p = gen_kwargs.get("top_p", 0.9) if do_sample else 1.0
        max_new_tokens = gen_kwargs.get("max_new_tokens", 128)
        seed = gen_kwargs.get("seed", None)
        
        dev = self.model.device
        g_noise = g_samp = None
        if seed is not None:
            g_noise = torch.Generator(device=dev).manual_seed(seed)
            g_samp = torch.Generator(device=dev).manual_seed(seed + 10_000)

        inp, is_batch, batch_size = self._build_inputs(image, prompt)
        cur_input_ids = inp["input_ids"]
        prompt_len = cur_input_ids.shape[1]

        # Initialize kwargs for clean image branch (run ViT on first step, use_cache, cache_position)
        kwargs_clean = {
            "attention_mask": inp["attention_mask"],
            "pixel_values": inp["pixel_values"].to(self.model.dtype),
            "image_grid_thw": inp["image_grid_thw"],
            "use_cache": True,
            "cache_position": torch.arange(0, prompt_len, device=dev),
        }

        # Initialize kwargs for distorted image branch (VCD)
        kwargs_cd = None
        if use_vcd:
            pv_cd = add_diffusion_noise_qwen2vl(
                inp["pixel_values"], vcd_noise_step, generator=g_noise
            ).to(self.model.dtype)
            kwargs_cd = {
                "attention_mask": inp["attention_mask"].clone(),
                "pixel_values": pv_cd,
                "image_grid_thw": inp["image_grid_thw"],
                "use_cache": True,
                "cache_position": torch.arange(0, prompt_len, device=dev),
            }

        # Track completion status for each sample in the batch
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=dev)
        gen_tokens = [[] for _ in range(batch_size)]

        # Autoregressive generation loop up to max_new_tokens (128)
        for _ in range(max_new_tokens):
            # 1. Forward clean branch
            inputs_clean = self.model.prepare_inputs_for_generation(cur_input_ids, **kwargs_clean)
            outputs_clean = self.model(**inputs_clean, return_dict=True)
            l = outputs_clean.logits[:, -1, :].float()
            kwargs_clean = self.model._update_model_kwargs_for_generation(outputs_clean, kwargs_clean)
            if hasattr(outputs_clean, "rope_deltas") and outputs_clean.rope_deltas is not None:
                kwargs_clean["rope_deltas"] = outputs_clean.rope_deltas

            # 2. Forward VCD branch
            if use_vcd:
                inputs_cd = self.model.prepare_inputs_for_generation(cur_input_ids, **kwargs_cd)
                outputs_cd = self.model(**inputs_cd, return_dict=True)
                l_cd = outputs_cd.logits[:, -1, :].float()
                kwargs_cd = self.model._update_model_kwargs_for_generation(outputs_cd, kwargs_cd)
                if hasattr(outputs_cd, "rope_deltas") and outputs_cd.rope_deltas is not None:
                    kwargs_cd["rope_deltas"] = outputs_cd.rope_deltas
                scores = vcd_contrast(l, l_cd, vcd_alpha, vcd_beta)
            else:
                scores = l

            # 3. Next token selection
            nxt = select_token(scores, do_sample=do_sample, temperature=temperature,
                               top_p=top_p, generator=g_samp)
            if nxt.ndim == 0:
                nxt = nxt.unsqueeze(0)

            # Record generated tokens for active sequences
            for idx in range(batch_size):
                if unfinished[idx]:
                    tok = int(nxt[idx])
                    if tok in self.eos_ids:
                        unfinished[idx] = False
                    else:
                        gen_tokens[idx].append(tok)

            # Early exit if all sequences in batch reached EOS
            if not unfinished.any():
                break

            cur_input_ids = torch.cat([cur_input_ids, nxt[:, None]], dim=1)

        # Decode generated token sequences
        captions = [
            self.processor.tokenizer.decode(t, skip_special_tokens=True).strip()
            for t in gen_tokens
        ]
        return captions if is_batch else captions[0]