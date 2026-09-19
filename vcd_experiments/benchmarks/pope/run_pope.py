import os
import sys
import json
import yaml
import random
import argparse
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np

# Ensure vcd_experiments root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VCD_EXP_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if VCD_EXP_DIR not in sys.path:
    sys.path.insert(0, VCD_EXP_DIR)

from benchmarks.pope.eval_pope import evaluate_predictions

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path: str = None) -> dict:
    search_paths = []
    if config_path:
        search_paths.append(config_path)
    
    search_paths.extend([
        os.path.join(VCD_EXP_DIR, "configs", "data_paths_kaggle.yaml"),
        os.path.join(VCD_EXP_DIR, "configs", "data_paths.yaml"),
        "configs/data_paths_kaggle.yaml",
        "configs/data_paths.yaml"
    ])
    
    for path in search_paths:
        if os.path.isfile(path):
            print(f"[Config] Loading data configuration from: {path}")
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
                
    raise FileNotFoundError(f"Could not find valid config file in search paths: {search_paths}")

def find_annotation_file(annotation_dir: str, split: str) -> str:
    """
    Finds the POPE annotation JSON file for a given split, checking multiple possible locations.
    """
    candidates = [
        os.path.join(annotation_dir, f"coco_pope_{split}.json"),
        os.path.join(annotation_dir, f"coco_pope_{split}.jsonl"),
        # Fallback to repo's experiments/data/POPE/coco/
        os.path.join(VCD_EXP_DIR, "..", "experiments", "data", "POPE", "coco", f"coco_pope_{split}.json"),
        os.path.join(VCD_EXP_DIR, "data", "POPE", "coco", f"coco_pope_{split}.json"),
        f"/kaggle/working/VCD/experiments/data/POPE/coco/coco_pope_{split}.json",
        f"/kaggle/input/pope/coco/coco_pope_{split}.json",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Cannot find annotation file for split '{split}'. Checked: {candidates}")

def find_image_file(image_dir: str, image_name: str) -> str:
    """
    Resolves image file path, checking subfolders if necessary.
    """
    candidates = [
        os.path.join(image_dir, image_name),
        os.path.join(image_dir, "val2014", image_name),
        f"/kaggle/input/coco-2014-dataset/val2014/{image_name}",
        f"/kaggle/input/coco2014/val2014/{image_name}",
        f"/kaggle/input/val2014/{image_name}",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return os.path.join(image_dir, image_name)

def run_single_split(
    model,
    model_type: str,
    split: str,
    anno_file: str,
    image_dir: str,
    output_dir: str,
    args: argparse.Namespace,
    run_timestamp: str,
    model_ckpt: str
) -> dict:
    """
    Runs POPE evaluation on a single split.
    """
    print(f"\n{'=' * 60}")
    print(f" Starting POPE Evaluation: Split='{split}' | Model='{model_type}' | VCD={args.use_vcd}")
    print(f" Annotation: {anno_file}")
    print(f"{'=' * 60}")

    with open(anno_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    if lines and lines[0].startswith("["):
        with open(anno_file, "r", encoding="utf-8") as f:
            pope_items = json.load(f)
    else:
        pope_items = [json.loads(line) for line in lines]
        
    if args.limit and args.limit > 0:
        pope_items = pope_items[:args.limit]
        print(f"[Notice] Limiting execution to first {args.limit} samples.")

    os.makedirs(output_dir, exist_ok=True)
    raw_outputs_file = os.path.join(output_dir, "raw_outputs.jsonl")
    run_config_file = os.path.join(output_dir, "run_config.json")
    metrics_file = os.path.join(output_dir, "metrics.json")

    # Record run config
    config_dict = {
        "model": model_type,
        "model_checkpoint": model_ckpt,
        "split": split,
        "use_vcd": args.use_vcd,
        "vcd_noise_step": args.noise_step if args.use_vcd else None,
        "vcd_alpha": args.cd_alpha if args.use_vcd else None,
        "vcd_beta": args.cd_beta if args.use_vcd else None,
        "max_new_tokens": 6,          # MANDATORY: 6 tokens
        "do_sample": False,           # MANDATORY: Greedy decoding
        "temperature": 0.0,
        "seed": args.seed,
        "timestamp": run_timestamp,
        "prompt_suffix": "Please answer with yes or no." if model_type == "qwen2vl" else None,
        "total_samples": len(pope_items)
    }
    with open(run_config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)

    results = []
    with open(raw_outputs_file, "w", encoding="utf-8") as f_out:
        pbar = tqdm(pope_items, desc=f"POPE {split} ({model_type})")
        for idx, item in enumerate(pbar):
            q_id = item.get("question_id", idx)
            img_name = item["image"]
            question = item["text"]
            label = item["label"]

            # Prompt formatting per model specification
            if model_type == "qwen2vl":
                # MANDATORY prompt suffix for QwenVL
                prompt = f"{question} Please answer with yes or no."
            else:
                prompt = question

            img_path = find_image_file(image_dir, img_name)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

            image = Image.open(img_path).convert("RGB")

            # Generation call with mandatory constraints (max_new_tokens=6, do_sample=False)
            if args.use_vcd:
                pred = model.generate(
                    prompt=prompt,
                    image=image,
                    use_vcd=True,
                    vcd_noise_step=args.noise_step,
                    vcd_alpha=args.cd_alpha,
                    vcd_beta=args.cd_beta,
                    max_new_tokens=6,
                    do_sample=False,
                    temperature=0.0
                )
            else:
                pred = model.generate(
                    prompt=prompt,
                    image=image,
                    use_vcd=False,
                    max_new_tokens=6,
                    do_sample=False,
                    temperature=0.0
                )

            record = {
                "question_id": q_id,
                "image": img_name,
                "question": question,
                "prompt": prompt,
                "label": label,
                "pred": pred,
                "text": pred,    # Compatibility with VCD eval_pope.py
                "answer": pred  # Compatibility with VQA eval standards
            }
            results.append(record)
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"\n[Done] Split '{split}' completed. Raw outputs saved to: {raw_outputs_file}")

    # Compute and save metrics
    metrics = evaluate_predictions(results)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("-" * 45)
    print(f" Split Summary: {split.upper()}")
    print("-" * 45)
    print(f"  Accuracy:    {metrics['Accuracy']}%")
    print(f"  Precision:   {metrics['Precision']}%")
    print(f"  Recall:      {metrics['Recall']}%")
    print(f"  F1-Score:    {metrics['F1']}%")
    print(f"  Yes Ratio:   {metrics['Yes_ratio']}%")
    print(f"  Total:       {metrics['Total']}")
    print(f"  Unknown:     {metrics['Unknown']}")
    print("-" * 45)

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Run POPE Benchmark with Visual Contrastive Decoding (VCD)")
    parser.add_argument("--model", type=str, choices=["llava", "qwen2vl"], required=True,
                        help="Target VLM to evaluate ('llava' or 'qwen2vl')")
    parser.add_argument("--split", type=str, choices=["random", "popular", "adversarial", "all"], default="all",
                        help="POPE split to evaluate ('random', 'popular', 'adversarial', or 'all')")
    parser.add_argument("--use_vcd", action="store_true", default=True,
                        help="Enable Visual Contrastive Decoding (default: True)")
    parser.add_argument("--no_vcd", action="store_false", dest="use_vcd",
                        help="Disable VCD to run standard baseline")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to YAML configuration file")
    parser.add_argument("--noise_step", type=int, default=500,
                        help="Diffusion noise step for VCD (default: 500)")
    parser.add_argument("--cd_alpha", type=float, default=1.0,
                        help="VCD contrastive penalty weight alpha (default: 1.0)")
    parser.add_argument("--cd_beta", type=float, default=0.1,
                        help="VCD adaptive plausibility cutoff beta (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples for rapid debugging")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Root output directory to save results")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to place model on ('auto', 'cuda:0', etc.)")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config_path)

    image_dir = config["coco_val2014_images"]
    annotation_dir = config["pope_coco_annotation_dir"]

    # Select model checkpoint
    if args.model == "llava":
        from models.llava_wrapper import LLaVAVCDWrapper
        model_ckpt = config["models"]["llava_1_5_7b_ckpt"]
        print(f"\n[Model] Initializing LLaVA VCD Wrapper from: {model_ckpt}")
        model = LLaVAVCDWrapper(model_path=model_ckpt, device=args.device, dtype=torch.bfloat16)
    elif args.model == "qwen2vl":
        from models.qwen2vl_wrapper import Qwen2VLVCDWrapper
        model_ckpt = config["models"]["qwen2vl_7b_instruct_ckpt"]
        print(f"\n[Model] Initializing Qwen2-VL VCD Wrapper from: {model_ckpt}")
        model = Qwen2VLVCDWrapper(model_path=model_ckpt, device=args.device, dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "vcd" if args.use_vcd else "baseline"
    
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        base_output_dir = os.path.join(VCD_EXP_DIR, "results", f"{args.model}_pope_{mode_str}_{run_timestamp}")

    splits_to_run = ["random", "popular", "adversarial"] if args.split == "all" else [args.split]
    summary_metrics = {}

    for split in splits_to_run:
        anno_file = find_annotation_file(annotation_dir, split)
        split_output_dir = os.path.join(base_output_dir, split) if len(splits_to_run) > 1 else base_output_dir
        
        metrics = run_single_split(
            model=model,
            model_type=args.model,
            split=split,
            anno_file=anno_file,
            image_dir=image_dir,
            output_dir=split_output_dir,
            args=args,
            run_timestamp=run_timestamp,
            model_ckpt=model_ckpt
        )
        summary_metrics[split] = metrics

    # Save overall multi-split summary
    if len(splits_to_run) > 1:
        summary_file = os.path.join(base_output_dir, "summary_all_splits.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=4)
            
        print("\n" + "=" * 65)
        print(f" ALL SPLITS OVERVIEW ({args.model.upper()} - {mode_str.upper()})")
        print("=" * 65)
        print(f"{'Split':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<10} {'Yes%':<10}")
        print("-" * 65)
        for s, m in summary_metrics.items():
            print(f"{s:<15} {m['Accuracy']:<12.2f} {m['Precision']:<12.2f} {m['Recall']:<12.2f} {m['F1']:<10.2f} {m['Yes_ratio']:<10.2f}")
        print("=" * 65)
        print(f"Overall summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
