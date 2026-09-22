import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from vcd_core.vcd_add_noise import add_diffusion_noise_qwen2vl
from vcd_core.vcd_decoding import vcd_contrast, select_token
from PIL import Image

class Qwen2VLVCDWrapper:
    def __init__(self, model_path, device="auto", dtype=torch.bfloat16):
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        load_kwargs = {
            "torch_dtype": dtype,
            "attn_implementation": "eager"
        }
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = device
            
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            **load_kwargs
        ).eval()
        
        if hasattr(self.model, "device"):
            self.device = self.model.device
        else:
            self.device = next(self.model.parameters()).device
            
        tok = self.processor.tokenizer
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        eos_set = set(eos if isinstance(eos, (list, tuple)) else ([eos] if eos is not None else []))
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None:
            eos_set.add(im_end_id)
        if tok.eos_token_id is not None:
            eos_set.add(tok.eos_token_id)
        self.eos_ids = {e for e in eos_set if e is not None}

    def _build_inputs(self, image, prompt):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=[text], images=[image.convert("RGB")], return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        if "mm_token_type_ids" not in inp and hasattr(self.model.config, "image_token_id"):
            inp["mm_token_type_ids"] = (inp["input_ids"] == self.model.config.image_token_id).int()
        return inp

    def _next_logits(self, input_ids, attn, mm, pv, thw):
        kw = dict(input_ids=input_ids, attention_mask=attn, pixel_values=pv,
                  image_grid_thw=thw, use_cache=False, logits_to_keep=1)
        if mm is not None:
            kw["mm_token_type_ids"] = mm
        return self.model(**kw).logits[:, -1, :].float()

    @torch.inference_mode()
    def generate(self, prompt, image: Image.Image, use_vcd=False, vcd_noise_step=500, vcd_alpha=1.0, vcd_beta=0.1, **gen_kwargs):
        do_sample = gen_kwargs.get("do_sample", False)
        temperature = gen_kwargs.get("temperature", 0.7) if do_sample else 1.0
        top_p = gen_kwargs.get("top_p", 0.9) if do_sample else 1.0
        max_new_tokens = gen_kwargs.get("max_new_tokens", 128)
        seed = gen_kwargs.get("seed", None)
        
        dev = self.device
        g_noise = g_samp = None
        if seed is not None:
            g_noise = torch.Generator(device=dev).manual_seed(seed)
            g_samp = torch.Generator(device=dev).manual_seed(seed + 10_000)

        inp = self._build_inputs(image, prompt)
        input_ids, attn = inp["input_ids"], inp["attention_mask"]
        mm = inp.get("mm_token_type_ids")
        thw = inp["image_grid_thw"]
        pv = inp["pixel_values"].to(self.model.dtype)
        
        if not use_vcd:
            # Baseline standard generation
            gen_args = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "eos_token_id": list(self.eos_ids) if self.eos_ids else None
            }
            if do_sample:
                gen_args["temperature"] = temperature
                gen_args["top_p"] = top_p
            
            # Use direct model generate for baseline efficiency
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                pixel_values=pv,
                image_grid_thw=thw,
                **gen_args
            )
            prompt_len = input_ids.shape[1]
            return self.processor.tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True).strip()

        # VCD generation loop
        pv_cd = add_diffusion_noise_qwen2vl(
            inp["pixel_values"], vcd_noise_step, generator=g_noise
        ).to(self.model.dtype)
            
        prompt_len = input_ids.shape[1]
        
        for _ in range(max_new_tokens):
            l = self._next_logits(input_ids, attn, mm, pv, thw)
            l_cd = self._next_logits(input_ids, attn, mm, pv_cd, thw)
            scores = vcd_contrast(l, l_cd, vcd_alpha, vcd_beta)
                
            nxt = select_token(scores, do_sample=do_sample, temperature=temperature,
                               top_p=top_p, generator=g_samp)
                               
            input_ids = torch.cat([input_ids, nxt[:, None]], dim=1)
            attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=1)
            if mm is not None:
                mm = torch.cat([mm, torch.zeros_like(mm[:, :1])], dim=1)
                
            if int(nxt) in self.eos_ids:
                break
                
        gen_ids = input_ids[0, prompt_len:]
        text = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text
