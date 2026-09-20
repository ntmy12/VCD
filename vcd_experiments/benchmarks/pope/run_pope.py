import argparse
import json
import os
import yaml
import torch
from tqdm import tqdm
from PIL import Image
from datetime import datetime

# Import custom LLaVA wrapper
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.llava_wrapper import LLaVAVCDWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Run POPE benchmark with LLaVA and VCD")
    parser.add_argument("--config", type=str, default="configs/data_paths.yaml", help="Path to data config")
    parser.add_argument("--split", type=str, choices=["random", "popular", "adversarial"], required=True, help="POPE split to run")
    parser.add_argument("--use_vcd", action="store_true", help="Enable Visual Contrastive Decoding")
    parser.add_argument("--vcd_noise_step", type=int, default=500, help="Noise step for VCD")
    parser.add_argument("--vcd_alpha", type=float, default=1.0, help="Alpha parameter for VCD")
    parser.add_argument("--vcd_beta", type=float, default=0.1, help="Beta parameter for VCD")
    parser.add_argument("--output_dir", type=str, default="results/", help="Directory to save results")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load config paths
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    pope_dir = config['pope_coco_annotation_dir']
    images_dir = config['coco_val2014_images']
    model_path = config['models']['llava_1_5_7b_ckpt']
    
    # 2. Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = "vcd" if args.use_vcd else "baseline"
    run_name = f"llava_pope_{args.split}_{method_name}_{timestamp}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # 3. Save run configuration
    run_config = vars(args)
    with open(os.path.join(out_dir, "run_config.json"), 'w') as f:
        json.dump(run_config, f, indent=4)
        
    # 4. Load dataset
    pope_file = os.path.join(pope_dir, f"coco_pope_{args.split}.json")
    dataset = []
    with open(pope_file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
            
    print(f"Loaded {len(dataset)} examples from {pope_file}")
    
    # 5. Load model
    print(f"Loading model from {model_path}...")
    model = LLaVAVCDWrapper(model_path, device="cuda:0", dtype=torch.bfloat16)
    
    # 6. Run Inference
    out_file = os.path.join(out_dir, "raw_outputs.jsonl")
    print(f"Starting generation. Saving to {out_file}...")
    
    with open(out_file, 'w') as f_out:
        for item in tqdm(dataset):
            image_name = item['image']
            question = item['text']
            ground_truth = item['label']
            
            image_path = os.path.join(images_dir, image_name)
            
            # Default LLaVA prompt format
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            
            try:
                image = Image.open(image_path).convert('RGB')
                
                # Generate using the custom generate function
                output_text = model.generate(
                    prompt=prompt,
                    image=image,
                    use_vcd=args.use_vcd,
                    vcd_noise_step=args.vcd_noise_step,
                    vcd_alpha=args.vcd_alpha,
                    vcd_beta=args.vcd_beta,
                    max_new_tokens=10,  # POPE needs short yes/no answers
                    do_sample=False     # POPE usually uses greedy
                )
                
                # Extract text after ASSISTANT:
                if "ASSISTANT:" in output_text:
                    output_text = output_text.split("ASSISTANT:")[-1].strip()
                
                result = {
                    "image": image_name,
                    "question": question,
                    "ground_truth": ground_truth,
                    "generated_text": output_text
                }
                
                f_out.write(json.dumps(result) + '\n')
                f_out.flush()
                
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                
    print(f"Done! Raw outputs saved to {out_file}")
    print("Now run the evaluation script on this output file.")

if __name__ == "__main__":
    main()
