import os
import sys
import json
import pickle
import argparse
import glob

# Ensure line buffering on stdout/stderr for immediate log streaming on Kaggle / Colab
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VCD_EXP_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if VCD_EXP_DIR not in sys.path:
    sys.path.insert(0, VCD_EXP_DIR)

from chair import CHAIR, print_metrics, save_hallucinated_words

# Make CHAIR available in __main__ if loaded via another script
setattr(sys.modules.get('__main__', sys), 'CHAIR', CHAIR)

class _CHAIRUnpickler(pickle.Unpickler):
    """
    Robust Unpickler that maps '__main__.CHAIR' or 'chair.CHAIR' to the loaded CHAIR class
    regardless of which script is the top-level __main__.
    """
    def find_class(self, module, name):
        if name == "CHAIR":
            return CHAIR
        return super().find_class(module, name)

def resolve_chair_cache(cache_path: str = None) -> str:
    candidates = []
    if cache_path:
        candidates.append(cache_path)
    candidates.extend([
        os.path.join(CURRENT_DIR, "chair.pkl"),
        os.path.join(os.path.dirname(CURRENT_DIR), "chair", "chair.pkl"),
        os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR)), "benchmarks", "chair", "chair.pkl"),
        "/kaggle/working/VCD/vcd_experiments/benchmarks/chair/chair.pkl",
        "/kaggle/working/vcd/vcd_experiments/benchmarks/chair/chair.pkl",
        "/content/VCD/vcd_experiments/benchmarks/chair/chair.pkl",
        "benchmarks/chair/chair.pkl",
        "chair.pkl"
    ])
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return None

def evaluate_chair(results_file: str, cache_path: str = None, coco_path: str = None, output_metrics_file: str = None, save_details_file: str = None) -> dict:
    if not os.path.isfile(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
        
    resolved_cache = resolve_chair_cache(cache_path)

    evaluator = None
    if resolved_cache and os.path.isfile(resolved_cache):
        print(f"[CHAIR Eval] Loading evaluator cache from: {resolved_cache}", flush=True)
        try:
            with open(resolved_cache, "rb") as f:
                evaluator = _CHAIRUnpickler(f).load()
        except Exception:
            with open(resolved_cache, "rb") as f:
                evaluator = pickle.load(f)
    elif coco_path and os.path.isdir(coco_path):
        print(f"[CHAIR Eval] Generating evaluator from COCO annotations: {coco_path}", flush=True)
        evaluator = CHAIR(coco_path)
    else:
        raise FileNotFoundError(
            f"Could not locate chair.pkl cache (checked '{cache_path}' and default fallbacks) and --coco_path not supplied."
        )

    res = evaluator.compute_chair(results_file, image_id_key="image_id", caption_key="caption")
    metrics = res['overall_metrics']
    metrics["total_evaluated"] = len(res['sentences'])

    print_metrics(res)

    if output_metrics_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_metrics_file)), exist_ok=True)
        with open(output_metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"[CHAIR Eval] Metrics saved to: {output_metrics_file}\n", flush=True)

    if save_details_file:
        save_hallucinated_words(save_details_file, res)
        print(f"[CHAIR Eval] Detailed breakdown saved to: {save_details_file}\n", flush=True)

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate CHAIR metrics for image captioning (CHAIRs, CHAIRi, Recall, Caption Length)")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to directory containing raw_outputs.jsonl")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Direct path to raw_outputs.jsonl or directory containing it")
    parser.add_argument("--cache_path", type=str, default=None,
                        help="Path to chair.pkl cache file")
    parser.add_argument("--coco_path", type=str, default=None,
                        help="Directory with COCO annotation json files if cache missing")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional explicit path to save metrics.json")
    parser.add_argument("--save_details", action="store_true",
                        help="Save detailed sentence-level hallucinations json")
    args = parser.parse_args()

    results_file = None
    if args.results_file:
        candidate = os.path.abspath(args.results_file)
        if os.path.isdir(candidate):
            results_file = os.path.join(candidate, "raw_outputs.jsonl")
        else:
            results_file = candidate
    elif args.results_dir:
        results_file = os.path.join(os.path.abspath(args.results_dir), "raw_outputs.jsonl")
    else:
        # Auto-discover most recent raw_outputs.jsonl in results/
        pattern = os.path.join(VCD_EXP_DIR, "results", "**", "raw_outputs.jsonl")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            results_file = matches[0]
            print(f"[CHAIR Eval] Auto-detected latest results file: {results_file}", flush=True)
        else:
            raise ValueError("Must specify either --results_dir or --results_file (or have a completed run in results/)")

    if not os.path.isfile(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    target_dir = os.path.dirname(os.path.abspath(results_file))
    metrics_file = args.output_file or os.path.join(target_dir, "metrics.json")
    details_file = os.path.join(target_dir, "chair_details.json") if args.save_details else None

    evaluate_chair(
        results_file=results_file,
        cache_path=args.cache_path,
        coco_path=args.coco_path,
        output_metrics_file=metrics_file,
        save_details_file=details_file
    )

if __name__ == "__main__":
    main()
