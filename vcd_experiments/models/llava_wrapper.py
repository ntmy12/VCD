import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from vcd_core.vcd_add_noise import add_diffusion_noise
from vcd_core.vcd_decoding import apply_vcd_penalty
import copy

class LLaVAVCDWrapper:
    def __init__(self, model_path, device="auto", dtype=torch.bfloat16):
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device
        )
        self.model.eval()
        if hasattr(self.model, "device"):
            self.device = self.model.device
        else:
            self.device = next(self.model.parameters()).device

    @torch.inference_mode()
    def generate(self, prompt, image, use_vcd=False, vcd_noise_step=500, vcd_alpha=1.0, vcd_beta=0.1, **gen_kwargs):
        """
        Generates text given a prompt and an image. 
        If use_vcd is True, applies Visual Contrastive Decoding.
        """
        if "<image>" not in prompt:
            prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        if "pixel_values" in inputs and self.dtype is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].to(torch.long)

        prompt_len = inputs["input_ids"].shape[1]

        if not use_vcd:
            output_ids = self.model.generate(**inputs, **gen_kwargs)
            return self.processor.decode(output_ids[0, prompt_len:], skip_special_tokens=True)

        # ---- VCD Generation Loop ----
        # 1. Create Distorted Image
        pixel_values = inputs["pixel_values"]
        pixel_values_cd = add_diffusion_noise(pixel_values, vcd_noise_step)
        
        inputs_cd = copy.deepcopy(inputs)
        inputs_cd["pixel_values"] = pixel_values_cd
        
        # 2. Extract inputs for custom generation loop
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        attention_mask_cd = inputs_cd["attention_mask"].clone()
        
        max_new_tokens = gen_kwargs.get("max_new_tokens", 128)
        eos_token_id = getattr(self.model, "generation_config", None)
        eos_token_id = getattr(eos_token_id, "eos_token_id", None) if eos_token_id else None
        if eos_token_id is None and hasattr(self.model, "config"):
            eos_token_id = getattr(self.model.config, "eos_token_id", None)
        if eos_token_id is None and hasattr(self.model.config, "text_config"):
            eos_token_id = getattr(self.model.config.text_config, "eos_token_id", None)
        if eos_token_id is None and hasattr(self.processor, "tokenizer"):
            eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        elif eos_token_id is None:
            eos_token_id = []
            
        past_key_values = None
        past_key_values_cd = None

        # Custom auto-regressive generation loop
        for step in range(max_new_tokens):
            # Forward original
            outputs = self.model(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                attention_mask=attention_mask,
                pixel_values=pixel_values if past_key_values is None else None,
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # Forward distorted
            outputs_cd = self.model(
                input_ids=input_ids if past_key_values_cd is None else input_ids[:, -1:],
                attention_mask=attention_mask_cd,
                pixel_values=pixel_values_cd if past_key_values_cd is None else None,
                past_key_values=past_key_values_cd,
                use_cache=True
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]
            past_key_values_cd = outputs_cd.past_key_values
            
            # Apply VCD penalty
            cd_logits = apply_vcd_penalty(next_token_logits, next_token_logits_cd, vcd_alpha, vcd_beta)
            
            # Sampling or greedy logic
            do_sample = gen_kwargs.get("do_sample", False)
            if do_sample:
                temperature = gen_kwargs.get("temperature", 1.0)
                cd_logits = cd_logits / temperature
                probs = torch.nn.functional.softmax(cd_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(cd_logits, dim=-1, keepdim=True)
                
            input_ids = torch.cat([input_ids, next_token.to(input_ids.device)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=-1)
            attention_mask_cd = torch.cat([attention_mask_cd, torch.ones((attention_mask_cd.shape[0], 1), device=attention_mask_cd.device, dtype=attention_mask_cd.dtype)], dim=-1)
            
            if next_token.item() in eos_token_id:
                break
                
        return self.processor.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
