import os
import sys
import time
import json
import yaml
import glob
import random
import argparse
from datetime import datetime, timedelta
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np

# Ensure line buffering on stdout/stderr for immediate log streaming on Kaggle / Colab
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

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
        os.path.join(VCD_EXP_DIR, "configs", "data_paths_colab.yaml"),
        os.path.join(VCD_EXP_DIR, "configs", "data_paths_kaggle.yaml"),
        os.path.join(VCD_EXP_DIR, "configs", "data_paths.yaml"),
        "configs/data_paths_colab.yaml",
        "configs/data_paths_kaggle.yaml",
        "configs/data_paths.yaml"
    ])
    
    for path in search_paths:
        if os.path.isfile(path):
            print(f"[Config] Loading data configuration from: {path}", flush=True)
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
                
    raise FileNotFoundError(f"Could not find valid config file in search paths: {search_paths}")

def resolve_coco_image_dir(configured_dir: str, sample_image: str = "COCO_val2014_000000102421.jpg") -> str:
    """
    Auto-detects and validates the COCO val2014 image directory on Colab, Kaggle, or local environments.
    """
    candidates = [
        configured_dir,
        os.path.join(configured_dir, "val2014") if configured_dir else "",
        os.path.join(CURRENT_DIR, "coco_chair_images"),
        os.path.join(VCD_EXP_DIR, "coco_chair_images"),
        os.path.join(VCD_EXP_DIR, "data", "coco2014", "val2014"),
        os.path.join(VCD_EXP_DIR, "data", "val2014"),
        "/content/data/val2014",
        "/content/data",
        "/content/val2014",
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
                print(f"[Image Dir] Verified COCO val2014 directory: {cand}", flush=True)
                return os.path.abspath(cand)
            try:
                files = os.listdir(cand)
                if any(f.startswith("COCO_val2014_") for f in files[:20]):
                    print(f"[Image Dir] Verified COCO val2014 directory: {cand}", flush=True)
                    return os.path.abspath(cand)
            except Exception:
                pass

    for search_root in ["/content/data", "/content", "/kaggle/input"]:
        if os.path.isdir(search_root):
            print(f"[Image Dir] Searching {search_root} for COCO val2014 images...", flush=True)
            for root, dirs, files in os.walk(search_root):
                depth = root.count(os.sep) - search_root.count(os.sep)
                if depth > 4:
                    continue
                if sample_image in files or any(f.startswith("COCO_val2014_") for f in files[:10]):
                    print(f"[Image Dir] Successfully auto-detected image directory: {root}", flush=True)
                    return os.path.abspath(root)

    print(f"[Image Dir] Notice: Using configured path: {configured_dir}", flush=True)
    return configured_dir

def find_image_file(image_dir: str, image_name: str) -> str:
    direct = os.path.join(image_dir, image_name)
    if os.path.isfile(direct):
        return direct
    sub = os.path.join(image_dir, "val2014", image_name)
    if os.path.isfile(sub):
        return sub

    candidates = [
        f"/content/data/val2014/{image_name}",
        f"/content/data/{image_name}",
        f"/content/val2014/{image_name}",
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
    base = os.path.basename(file_name).split(".")[0]
    num_part = base.split("_")[-1]
    return int(num_part)

def select_samples(image_dir: str, seed: int = 2027, num_samples: int = 500, manifest_file: str = None) -> list:
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
                print(f"[Sampling] Using verified pre-computed manifest ({len(samples[:num_samples])} images): {m}", flush=True)
                return samples[:num_samples]
        except Exception as e:
            print(f"[Sampling Warning] Could not parse manifest {m}: {e}", flush=True)

    # Fallback to deterministic sampling from directory
    print(f"[Sampling] Manifest not found, deterministically sampling {num_samples} images with seed {seed} from {image_dir}...", flush=True)
    all_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if len(all_files) == 0:
        raise FileNotFoundError(f"No image files found in {image_dir}")
        
    chosen = random.Random(seed).sample(all_files, min(num_samples, len(all_files)))
    samples = [{"image_id": extract_image_id(fn), "file_name": fn} for fn in chosen]
    return samples

def resolve_resume_target(resume: bool, resume_from: str, output_dir: str, model: str, mode_str: str, run_timestamp: str, vcd_exp_dir: str):
    """
    Intelligently resolves target output directory and raw_outputs.jsonl file.
    """
    results_root = os.path.join(vcd_exp_dir, "results")

    # 1. Explicit resume_from path provided
    if resume_from:
        candidate = os.path.abspath(resume_from)
        if os.path.isfile(candidate):
            return os.path.dirname(candidate), candidate
        elif os.path.isdir(candidate):
            target_file = os.path.join(candidate, "raw_outputs.jsonl")
            return candidate, target_file
        else:
            alt = os.path.join(results_root, resume_from)
            if os.path.isfile(alt):
                return os.path.dirname(alt), alt
            elif os.path.isdir(alt):
                return alt, os.path.join(alt, "raw_outputs.jsonl")
            raise FileNotFoundError(f"--resume_from path not found: {resume_from}")

    # 2. --resume flag given without explicit path: find most recent run
    if resume:
        if output_dir and os.path.isdir(output_dir):
            target_file = os.path.join(output_dir, "raw_outputs.jsonl")
            return os.path.abspath(output_dir), target_file

        if os.path.isdir(results_root):
            candidates = []
            for d in os.listdir(results_root):
                full_d = os.path.join(results_root, d)
                if os.path.isdir(full_d):
                    raw_f = os.path.join(full_d, "raw_outputs.jsonl")
                    if os.path.isfile(raw_f):
                        mtime = os.path.getmtime(raw_f)
                        score = 0
                        if model.lower() in d.lower():
                            score += 2
                        if "chair" in d.lower():
                            score += 2
                        if mode_str.lower() in d.lower():
                            score += 1
                        candidates.append((score, mtime, full_d, raw_f))

            if candidates:
                candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                best_dir, best_file = candidates[0][2], candidates[0][3]
                print(f"[Resume] Auto-detected latest matching run: {best_dir}", flush=True)
                return best_dir, best_file
            else:
                print(f"[Resume Notice] No previous run found with raw_outputs.jsonl in {results_root}. Starting fresh run.", flush=True)

    # 3. Custom output_dir or fallback to timestamped directory
    if output_dir:
        resolved_dir = os.path.abspath(output_dir)
    else:
        resolved_dir = os.path.join(results_root, f"{model}_chair_{mode_str}_{run_timestamp}")

    return resolved_dir, os.path.join(resolved_dir, "raw_outputs.jsonl")

def main():
    parser = argparse.ArgumentParser(description="Run CHAIR Benchmark with Visual Contrastive Decoding (VCD)")
    parser.add_argument("--model", type=str, default="llava", choices=["llava", "qwen2vl"],
                        help="Target VLM to evaluate ('llava' or 'qwen2vl', default: 'llava')")
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
                        help="Number of images to evaluate (default: 500; set to 10 for latency profiling)")
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
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Auto-resume from the latest matching run in results directory")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Direct path to raw_outputs.jsonl or result directory to resume from")
    parser.add_argument("--log_interval", type=int, default=5,
                        help="Number of samples between explicit newline progress logs (default: 5)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to place model on ('auto', 'cuda:0', etc.)")
    parser.add_argument("--dtype", type=str, choices=["auto", "bf16", "fp16", "16", "fp32"], default="fp16",
                        help="Model precision (16-bit: 'fp16', 'bf16', 'auto')")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config_path)

    image_dir = resolve_coco_image_dir(config.get("coco_val2014_images", ""))
    manifest_file = config.get("chair_manifest", None)

    # Select the images (default 500 or user specified, e.g. 10)
    samples = select_samples(
        image_dir=image_dir,
        seed=args.seed,
        num_samples=args.num_samples,
        manifest_file=manifest_file
    )

    if args.limit and args.limit > 0:
        samples = samples[:args.limit]
        print(f"[Notice] Limiting execution to first {args.limit} samples.", flush=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "vcd" if args.use_vcd else "baseline"

    # Resolve output directory and raw_outputs file
    output_dir, raw_outputs_file = resolve_resume_target(
        resume=args.resume,
        resume_from=args.resume_from,
        output_dir=args.output_dir,
        model=args.model,
        mode_str=mode_str,
        run_timestamp=run_timestamp,
        vcd_exp_dir=VCD_EXP_DIR
    )
    os.makedirs(output_dir, exist_ok=True)
    run_config_file = os.path.join(output_dir, "run_config.json")
    metrics_file = os.path.join(output_dir, "metrics.json")
    summary_file = os.path.join(output_dir, "summary_metrics.json")

    # Read existing completed records
    results = []
    completed_ids = set()
    sample_latencies = []
    if os.path.isfile(raw_outputs_file):
        try:
            with open(raw_outputs_file, "r", encoding="utf-8") as f_prev:
                for line in f_prev:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        results.append(record)
                        if "image_id" in record:
                            completed_ids.add(record["image_id"])
                        if "latency_s" in record:
                            sample_latencies.append(record["latency_s"])
            if completed_ids:
                print(f"[Resume] Successfully loaded {len(completed_ids)} completed samples from {raw_outputs_file}.", flush=True)
        except Exception as e:
            print(f"[Resume Warning] Could not parse existing results ({e}), starting fresh.", flush=True)
            results = []
            completed_ids = set()
            sample_latencies = []

    # Determine remaining samples
    remaining_samples = [s for s in samples if s["image_id"] not in completed_ids]

    print(f"\n{'=' * 65}", flush=True)
    print(f" CHAIR Caption Generation Setup: Model='{args.model}' | VCD={args.use_vcd}", flush=True)
    print(f" Target Samples: {len(samples)} | Already Done: {len(completed_ids)} | Remaining: {len(remaining_samples)}", flush=True)
    print(f" Seed: {args.seed} | Max Tokens: {args.max_new_tokens} | Log Interval: {args.log_interval}", flush=True)
    print(f" Output Directory: {output_dir}", flush=True)
    print(f"{'=' * 65}\n", flush=True)

    # Resolve model checkpoint
    if args.model == "llava":
        model_ckpt = config.get("models", {}).get("llava_1_5_7b_ckpt", "llava-hf/llava-1.5-7b-hf")
    elif args.model == "qwen2vl":
        model_ckpt = config.get("models", {}).get("qwen2vl_7b_instruct_ckpt", "Qwen/Qwen2-VL-7B-Instruct")
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Conditional Generation: only load model if there are samples remaining
    if len(remaining_samples) > 0:
        dtype = resolve_dtype(args.dtype)
        print(f"[Precision] Using resolved precision: {dtype} (requested: {args.dtype})", flush=True)

        # Initialize Model Wrapper
        if args.model == "llava":
            from models.llava_wrapper import LLaVAVCDWrapper
            print(f"[Model] Initializing LLaVA VCD Wrapper from: {model_ckpt}...", flush=True)
            model = LLaVAVCDWrapper(model_path=model_ckpt, device=args.device, dtype=dtype)
        elif args.model == "qwen2vl":
            from models.qwen2vl_wrapper import Qwen2VLVCDWrapper
            print(f"[Model] Initializing Qwen2-VL VCD Wrapper from: {model_ckpt}...", flush=True)
            model = Qwen2VLVCDWrapper(model_path=model_ckpt, device=args.device, dtype=dtype)

        # Save/update run config
        config_dict = {
            "model": args.model,
            "model_checkpoint": model_ckpt,
            "benchmark": "chair",
            "use_vcd": args.use_vcd,
            "vcd_noise_step": args.noise_step if args.use_vcd else None,
            "vcd_alpha": args.cd_alpha if args.use_vcd else None,
            "vcd_beta": args.cd_beta if args.use_vcd else None,
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

        write_mode = "a" if completed_ids else "w"
        loop_start_time = time.time()
        print(f"Starting caption generation for {len(remaining_samples)} remaining images...", flush=True)

        with open(raw_outputs_file, write_mode, encoding="utf-8") as f_out:
            pbar = tqdm(remaining_samples, desc=f"CHAIR ({args.model} - {mode_str})", mininterval=2.0)
            for gen_idx, item in enumerate(pbar):
                img_id = item["image_id"]
                img_name = item["file_name"]

                img_path = find_image_file(image_dir, img_name)
                if not os.path.isfile(img_path):
                    raise FileNotFoundError(f"Image not found: {img_path}")

                image = Image.open(img_path).convert("RGB")

                # Precise CUDA inference timing with torch.cuda.synchronize()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_start_time = time.perf_counter()

                if args.use_vcd:
                    pred = model.generate(
                        prompt=args.prompt,
                        image=image,
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
                    pred = model.generate(
                        prompt=args.prompt,
                        image=image,
                        use_vcd=False,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature
                    )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_end_time = time.perf_counter()
                step_time = step_end_time - step_start_time
                sample_latencies.append(step_time)

                record = {
                    "image_id": img_id,
                    "file_name": img_name,
                    "prompt": args.prompt,
                    "caption": pred,
                    "latency_s": round(step_time, 4)
                }
                results.append(record)
                completed_ids.add(img_id)
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                try:
                    os.fsync(f_out.fileno())
                except Exception:
                    pass

                # Real-time progress logging with explicit newline and flush for Kaggle / Colab
                total_done = len(samples) - len(remaining_samples) + (gen_idx + 1)
                is_periodic = ((gen_idx + 1) % args.log_interval == 0)
                is_first = (gen_idx == 0)
                is_last = (gen_idx == len(remaining_samples) - 1)

                if is_periodic or is_first or is_last:
                    elapsed = time.time() - loop_start_time
                    speed = (gen_idx + 1) / elapsed if elapsed > 0 else 0
                    sec_per_it = 1.0 / speed if speed > 0 else 0.0
                    remaining_count = len(remaining_samples) - (gen_idx + 1)
                    eta_sec = remaining_count * sec_per_it
                    eta_str = str(timedelta(seconds=int(eta_sec)))
                    pct = (total_done / len(samples)) * 100

                    log_msg = (
                        f"[CHAIR Progress] [{total_done}/{len(samples)}] ({pct:5.1f}%) | "
                        f"Step: {step_time:4.2f}s | Speed: {sec_per_it:4.2f}s/it | "
                        f"ETA: {eta_str} | ID: {img_id} ({img_name})\n"
                    )
                    sys.stdout.write(log_msg)
                    sys.stdout.flush()

        print(f"\n[Done] Caption generation completed. Raw outputs saved to: {raw_outputs_file}\n", flush=True)
    else:
        print(f"[Resume] All {len(samples)} samples are already completed! Skipping model loading & generation.", flush=True)
        print(f"[Resume] Proceeding directly to CHAIR evaluation on: {raw_outputs_file}\n", flush=True)

    # Compute overall timing statistics
    total_inference_time = sum(sample_latencies)
    avg_time_per_sample = (total_inference_time / len(sample_latencies)) if sample_latencies else 0.0

    print(f"\n{'=' * 65}", flush=True)
    print(f"⏱️ INFERENCE RUNTIME SUMMARY | Model: {args.model.upper()} | VCD: {args.use_vcd}", flush=True)
    print(f"  • Total Images Evaluated  : {len(results)} samples", flush=True)
    print(f"  • Total Inference Time    : {total_inference_time:.2f} seconds", flush=True)
    print(f"  • AVERAGE TIME / SAMPLE   : {avg_time_per_sample:.4f} seconds / sample", flush=True)
    if avg_time_per_sample > 0:
        print(f"  • Throughput              : {1.0 / avg_time_per_sample:.2f} samples / second", flush=True)
    print(f"{'=' * 65}\n", flush=True)

    # Run CHAIR evaluation
    print(f"[CHAIR] Running automatic metric evaluation...", flush=True)
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
    except Exception as e:
        print(f"\n[CHAIR Warning] Automatic evaluation encountered an issue: {e}", flush=True)
        metrics = {"total_evaluated": len(results)}

    # Attach timing metrics to metrics dictionary
    metrics["avg_time_per_sample_s"] = round(avg_time_per_sample, 4)
    metrics["total_inference_time_s"] = round(total_inference_time, 4)
    metrics["num_evaluated"] = len(results)
    metrics["samples_tested"] = len(samples)

    # Re-save metrics.json with timing metrics included
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Save summary_metrics.json for cross-run aggregation
    summary_data = {
        "model": args.model,
        "mode": mode_str,
        "use_vcd": args.use_vcd,
        "metrics": metrics,
        "overall": {
            "total_samples": len(results),
            "total_inference_time_s": round(total_inference_time, 4),
            "overall_avg_time_per_sample_s": round(avg_time_per_sample, 4)
        }
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    print("\n" + "=" * 65, flush=True)
    print(f" CHAIR BENCHMARK FINAL SUMMARY ({args.model.upper()} - {mode_str.upper()})", flush=True)
    print("=" * 65, flush=True)
    print(f"  CHAIRs (Sentence Hallucination Rate): {metrics.get('CHAIRs', 0.0):.2f}%", flush=True)
    print(f"  CHAIRi (Instance Hallucination Rate): {metrics.get('CHAIRi', 0.0):.2f}%", flush=True)
    print(f"  Recall (Ground-truth Objects Recall): {metrics.get('Recall', 0.0):.2f}%", flush=True)
    print(f"  Caption Length (Average Words):       {metrics.get('Caption_Length', 0.0):.2f} words", flush=True)
    print(f"  Total Images Evaluated:               {metrics.get('num_evaluated', len(samples))}", flush=True)
    print(f"  Total Inference Time:                 {total_inference_time:.2f} seconds", flush=True)
    print(f"  Average Runtime Per Sample:           {avg_time_per_sample:.4f} seconds / sample", flush=True)
    if avg_time_per_sample > 0:
        print(f"  Throughput:                           {1.0 / avg_time_per_sample:.2f} samples / second", flush=True)
    print("=" * 65, flush=True)
    print(f"Results and metrics saved to: {output_dir}\n", flush=True)

if __name__ == "__main__":
    main()
