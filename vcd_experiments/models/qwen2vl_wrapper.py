import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from vcd_core.vcd_add_noise import add_diffusion_noise
from vcd_core.vcd_decoding import apply_vcd_penalty
import copy
from PIL import Image

class Qwen2VLVCDWrapper:
    def __init__(self, model_path, device="auto", dtype=torch.bfloat16):
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
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
    def generate(self, prompt, image: Image.Image, use_vcd=False, vcd_noise_step=500, vcd_alpha=1.0, vcd_beta=0.1, **gen_kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(self.device)
        
        # We need to cast pixel_values to bfloat16
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            
        if not use_vcd:
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            return self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        # ---- VCD Generation Loop ----
        pixel_values = inputs["pixel_values"]
        pixel_values_cd = add_diffusion_noise(pixel_values, vcd_noise_step)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        attention_mask_cd = attention_mask.clone()
        image_grid_thw = inputs.get("image_grid_thw", None)
        
        max_new_tokens = gen_kwargs.get("max_new_tokens", 128)
        eos_token_id = self.model.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
            
        past_key_values = None
        past_key_values_cd = None

        new_token_length = 0
        
        for step in range(max_new_tokens):
            outputs = self.model(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                attention_mask=attention_mask,
                pixel_values=pixel_values if past_key_values is None else None,
                image_grid_thw=image_grid_thw if past_key_values is None else None,
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            outputs_cd = self.model(
                input_ids=input_ids if past_key_values_cd is None else input_ids[:, -1:],
                attention_mask=attention_mask_cd,
                pixel_values=pixel_values_cd if past_key_values_cd is None else None,
                image_grid_thw=image_grid_thw if past_key_values_cd is None else None,
                past_key_values=past_key_values_cd,
                use_cache=True
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]
            past_key_values_cd = outputs_cd.past_key_values
            
            cd_logits = apply_vcd_penalty(next_token_logits, next_token_logits_cd, vcd_alpha, vcd_beta)
            
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
            
            new_token_length += 1
            if eos_token_id is not None and next_token.item() in eos_token_id:
                break
                
        generated_ids_trimmed = input_ids[:, -new_token_length:]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
