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
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from benchmarks.chair.eval_chair import evaluate_chair

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

def resolve_coco_image_dir(configured_dir: str, sample_image: str = "COCO_val2014_000000102421.jpg") -> str:
    """
    Auto-detects and validates the COCO val2014 image directory on Kaggle or local environments.
    """
    candidates = [
        configured_dir,
        os.path.join(configured_dir, "val2014") if configured_dir else "",
        os.path.join(CURRENT_DIR, "coco_chair_images"),
        os.path.join(VCD_EXP_DIR, "coco_chair_images"),
        os.path.join(VCD_EXP_DIR, "data", "coco2014", "val2014"),
        os.path.join(VCD_EXP_DIR, "data", "val2014"),
        "/kaggle/input/datasets/biminhco/val2014/val2014",
        "/kaggle/input/datasets/biminhco/val2014",
        "/kaggle/input/biminhco/val2014/val2014",
        "/kaggle/input/biminhco/val2014",
        "/kaggle/input/val2014/val2014",
        "/kaggle/input/val2014",
        "/kaggle/input/coco-2014-dataset/val2014",
        "/kaggle/input/coco2014/val2014",
    ]
    for cand in candidates:
        if cand and os.path.isdir(cand):
            if os.path.isfile(os.path.join(cand, sample_image)):
                print(f"[Image Dir] Verified COCO val2014 directory: {cand}")
                return os.path.abspath(cand)
            try:
                files = os.listdir(cand)
                if any(f.startswith("COCO_val2014_") for f in files[:20]):
                    print(f"[Image Dir] Verified COCO val2014 directory: {cand}")
                    return os.path.abspath(cand)
            except Exception:
                pass

    if os.path.isdir("/kaggle/input"):
        print("[Image Dir] Searching /kaggle/input for COCO val2014 images...")
        for root, dirs, files in os.walk("/kaggle/input"):
            depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
            if depth > 5:
                continue
            if sample_image in files or any(f.startswith("COCO_val2014_") for f in files[:10]):
                print(f"[Image Dir] Successfully auto-detected image directory: {root}")
                return os.path.abspath(root)

    print(f"[Image Dir] Notice: Using configured path: {configured_dir}")
    return configured_dir

def find_image_file(image_dir: str, image_name: str) -> str:
    direct = os.path.join(image_dir, image_name)
    if os.path.isfile(direct):
        return direct
    sub = os.path.join(image_dir, "val2014", image_name)
    if os.path.isfile(sub):
        return sub

    candidates = [
        f"/kaggle/input/datasets/biminhco/val2014/val2014/{image_name}",
        f"/kaggle/input/datasets/biminhco/val2014/{image_name}",
        f"/kaggle/input/biminhco/val2014/val2014/{image_name}",
        f"/kaggle/input/val2014/val2014/{image_name}",
        f"/kaggle/input/val2014/{image_name}",
        f"/kaggle/input/coco-2014-dataset/val2014/{image_name}",
        f"/kaggle/input/coco2014/val2014/{image_name}",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return direct

def resolve_dtype(dtype_str: str = "auto") -> torch.dtype:
    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    elif dtype_str in ("fp16", "float16", "16"):
        return torch.float16
    elif dtype_str in ("fp32", "float32"):
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def extract_image_id(file_name: str) -> int:
    # Handles COCO_val2014_000000102421.jpg -> 102421
    base = os.path.basename(file_name).split(".")[0]
    num_part = base.split("_")[-1]
    return int(num_part)

def select_samples(image_dir: str, seed: int = 2027, num_samples: int = 500, manifest_file: str = None) -> list:
    """
    Loads samples from manifest file if available, or deterministically samples from image_dir.
    """
    candidates = []
    if manifest_file and os.path.isfile(manifest_file):
        candidates.append(manifest_file)
    default_manifest = os.path.join(CURRENT_DIR, f"selected_chair_val2014_seed{seed}.json")
    if os.path.isfile(default_manifest):
        candidates.append(default_manifest)

    for m in candidates:
        try:
            with open(m, "r", encoding="utf-8") as f:
                data = json.load(f)
            samples = data.get("samples", data)
            if isinstance(samples, list) and len(samples) >= num_samples:
                print(f"[Sampling] Using verified pre-computed manifest ({len(samples[:num_samples])} images): {m}")
                return samples[:num_samples]
        except Exception as e:
            print(f"[Sampling Warning] Could not parse manifest {m}: {e}")

    # Fallback to deterministic sampling from directory
    print(f"[Sampling] Manifest not found, deterministically sampling {num_samples} images with seed {seed} from {image_dir}...")
    all_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if len(all_files) == 0:
        raise FileNotFoundError(f"No image files found in {image_dir}")
        
    chosen = random.Random(seed).sample(all_files, min(num_samples, len(all_files)))
    samples = [{"image_id": extract_image_id(fn), "file_name": fn} for fn in chosen]
    return samples

def main():
    parser = argparse.ArgumentParser(description="Run CHAIR Benchmark with Visual Contrastive Decoding (VCD)")
    parser.add_argument("--model", type=str, default="qwen2vl", choices=["qwen2vl"],
                        help="Target VLM to evaluate (default: 'qwen2vl')")
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
    parser.add_argument("--seed", type=int, default=2027,
                        help="Random seed for image selection and reproducibility (default: 2027)")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of images to evaluate (default: 500)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum new tokens generated per image (default: 128)")
    parser.add_argument("--prompt", type=str, default="Describe this image.",
                        help="Prompt fed to model (default: 'Describe this image.')")
    parser.add_argument("--do_sample", action="store_true", default=False,
                        help="Enable sampling decoding (default: False for greedy decoding)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for generation (default: 0.0 for greedy)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples for rapid debugging")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Root output directory to save results")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to place model on ('auto', 'cuda:0', etc.)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for parallel caption generation (default: 1; recommended: 4 or 8 on RTX 6000)")
    parser.add_argument("--dtype", type=str, choices=["auto", "bf16", "fp16", "16", "fp32"], default="bf16",
                        help="Model precision (16-bit: 'bf16' on RTX 6000 Ada / A6000, 'fp16', 'auto')")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config_path)

    image_dir = resolve_coco_image_dir(config.get("coco_val2014_images", ""))
    manifest_file = config.get("chair_manifest", None)

    # Select the 500 images
    samples = select_samples(
        image_dir=image_dir,
        seed=args.seed,
        num_samples=args.num_samples,
        manifest_file=manifest_file
    )

    if args.limit and args.limit > 0:
        samples = samples[:args.limit]
        print(f"[Notice] Limiting execution to first {args.limit} samples.")

    # Precision (16-bit)
    dtype = resolve_dtype(args.dtype)
    print(f"[Precision] Using resolved 16-bit precision: {dtype} (requested: {args.dtype})")

    # Initialize Qwen2-VL Model Wrapper
    from models.qwen2vl_wrapper import Qwen2VLVCDWrapper
    model_ckpt = config.get("models", {}).get("qwen2vl_7b_instruct_ckpt", "Qwen/Qwen2-VL-7B-Instruct")
    print(f"\n[Model] Initializing Qwen2-VL VCD Wrapper (16-bit) from: {model_ckpt}")
    model = Qwen2VLVCDWrapper(model_path=model_ckpt, device=args.device, dtype=dtype)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "vcd" if args.use_vcd else "baseline"

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(VCD_EXP_DIR, "results", f"{args.model}_chair_{mode_str}_{run_timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    raw_outputs_file = os.path.join(output_dir, "raw_outputs.jsonl")
    run_config_file = os.path.join(output_dir, "run_config.json")
    metrics_file = os.path.join(output_dir, "metrics.json")

    # Automatic Resume Support
    results = []
    completed_ids = set()
    if os.path.isfile(raw_outputs_file):
        try:
            with open(raw_outputs_file, "r", encoding="utf-8") as f_prev:
                for line in f_prev:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        results.append(record)
                        completed_ids.add(record.get("image_id"))
            if completed_ids:
                print(f"[Resume] Found {len(completed_ids)} completed samples in {raw_outputs_file}. Resuming from sample {len(completed_ids) + 1}...")
        except Exception as e:
            print(f"[Resume Warning] Could not parse existing results ({e}), starting fresh.")
            results = []
            completed_ids = set()

    # Log run config
    config_dict = {
        "model": args.model,
        "model_checkpoint": model_ckpt,
        "benchmark": "chair",
        "use_vcd": args.use_vcd,
        "vcd_noise_step": args.noise_step if args.use_vcd else None,
        "vcd_alpha": args.cd_alpha if args.use_vcd else None,
        "vcd_beta": args.cd_beta if args.use_vcd else None,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "prompt": args.prompt,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "seed": args.seed,
        "timestamp": run_timestamp,
        "total_samples": len(samples)
    }
    with open(run_config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)

    print(f"\n{'=' * 65}")
    print(f" Starting CHAIR Caption Generation: Model='{args.model}' | VCD={args.use_vcd}")
    print(f" Samples: {len(samples)} | Batch Size: {args.batch_size} | Seed: {args.seed} | Max Tokens: {args.max_new_tokens}")
    print(f" Prompt: \"{args.prompt}\"")
    print(f"{'=' * 65}")

    write_mode = "a" if completed_ids else "w"
    pending_samples = [s for s in samples if s["image_id"] not in completed_ids]

    with open(raw_outputs_file, write_mode, encoding="utf-8") as f_out:
        pbar = tqdm(total=len(samples), initial=len(completed_ids), desc=f"CHAIR ({args.model} - {mode_str})")
        batch_size = max(1, args.batch_size)

        for i in range(0, len(pending_samples), batch_size):
            chunk = pending_samples[i:i + batch_size]
            images = []
            valid_chunk = []
            for item in chunk:
                img_name = item["file_name"]
                img_path = find_image_file(image_dir, img_name)
                if not os.path.isfile(img_path):
                    raise FileNotFoundError(f"Image not found: {img_path}")
                images.append(Image.open(img_path).convert("RGB"))
                valid_chunk.append(item)

            input_images = images if len(images) > 1 else images[0]

            if args.use_vcd:
                preds = model.generate(
                    prompt=args.prompt,
                    image=input_images,
                    use_vcd=True,
                    vcd_noise_step=args.noise_step,
                    vcd_alpha=args.cd_alpha,
                    vcd_beta=args.cd_beta,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    seed=args.seed
                )
            else:
                preds = model.generate(
                    prompt=args.prompt,
                    image=input_images,
                    use_vcd=False,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature
                )

            if isinstance(preds, str):
                preds = [preds]

            for item, pred in zip(valid_chunk, preds):
                record = {
                    "image_id": item["image_id"],
                    "file_name": item["file_name"],
                    "prompt": args.prompt,
                    "caption": pred
                }
                results.append(record)
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            f_out.flush()
            pbar.update(len(valid_chunk))

        pbar.close()

    print(f"\n[Done] Caption generation completed. Raw outputs saved to: {raw_outputs_file}")

    # Run CHAIR evaluation
    print(f"\n[CHAIR] Running automatic metric evaluation...")
    cache_path = config.get("chair_eval_cache", None)
    if cache_path and not os.path.isabs(cache_path):
        cache_path = os.path.join(VCD_EXP_DIR, cache_path)

    try:
        metrics = evaluate_chair(
            results_file=raw_outputs_file,
            cache_path=cache_path,
            coco_path=config.get("chair_annotation_dir", None),
            output_metrics_file=metrics_file,
            save_details_file=os.path.join(output_dir, "chair_details.json")
        )

        print("\n" + "=" * 65)
        print(f" CHAIR BENCHMARK FINAL SUMMARY ({args.model.upper()} - {mode_str.upper()})")
        print("=" * 65)
        print(f"  CHAIRs (Sentence Hallucination Rate): {metrics['CHAIRs']:.2f}%")
        print(f"  CHAIRi (Instance Hallucination Rate): {metrics['CHAIRi']:.2f}%")
        print(f"  Recall (Ground-truth Objects Recall): {metrics['Recall']:.2f}%")
        print(f"  Caption Length (Average Words):       {metrics['Caption_Length']:.2f} words")
        print(f"  Total Images Evaluated:               {metrics.get('total_evaluated', len(samples))}")
        print("=" * 65)
        print(f"Results and metrics saved to: {output_dir}\n")
    except Exception as e:
        print(f"\n[CHAIR Warning] Automatic evaluation encountered an issue: {e}")
        print(f"You can rerun evaluation directly with:")
        print(f"  python benchmarks/chair/eval_chair.py --results_file {raw_outputs_file}\n")

if __name__ == "__main__":
    main()
