import os
import sys
import json
import pickle
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from chair import CHAIR, print_metrics, save_hallucinated_words

def evaluate_chair(results_file: str, cache_path: str = None, coco_path: str = None, output_metrics_file: str = None, save_details_file: str = None) -> dict:
    if not os.path.isfile(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
        
    if not cache_path:
        default_cache = os.path.join(CURRENT_DIR, "chair.pkl")
        cache_path = default_cache if os.path.isfile(default_cache) else None

    evaluator = None
    if cache_path and os.path.isfile(cache_path):
        print(f"[CHAIR Eval] Loading evaluator cache from: {cache_path}")
        with open(cache_path, "rb") as f:
            evaluator = pickle.load(f)
    elif coco_path and os.path.isdir(coco_path):
        print(f"[CHAIR Eval] Generating evaluator from COCO annotations: {coco_path}")
        evaluator = CHAIR(coco_path)
    else:
        raise FileNotFoundError(
            f"Could not locate chair.pkl cache (checked '{cache_path}') and --coco_path not supplied."
        )

    res = evaluator.compute_chair(results_file, image_id_key="image_id", caption_key="caption")
    metrics = res['overall_metrics']
    metrics["total_evaluated"] = len(res['sentences'])

    print_metrics(res)

    if output_metrics_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_metrics_file)), exist_ok=True)
        with open(output_metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"[CHAIR Eval] Metrics saved to: {output_metrics_file}")

    if save_details_file:
        save_hallucinated_words(save_details_file, res)
        print(f"[CHAIR Eval] Detailed breakdown saved to: {save_details_file}")

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate CHAIR metrics for image captioning (CHAIRs, CHAIRi, Recall, Caption Length)")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to directory containing raw_outputs.jsonl")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Direct path to raw_outputs.jsonl")
    parser.add_argument("--cache_path", type=str, default=None,
                        help="Path to chair.pkl cache file")
    parser.add_argument("--coco_path", type=str, default=None,
                        help="Directory with COCO annotation json files if cache missing")
    parser.add_argument("--save_details", action="store_true",
                        help="Save detailed sentence-level hallucinations json")
    args = parser.parse_args()

    if args.results_file:
        results_file = args.results_file
        metrics_file = os.path.join(os.path.dirname(results_file), "metrics.json")
        details_file = os.path.join(os.path.dirname(results_file), "chair_details.json") if args.save_details else None
    elif args.results_dir:
        results_file = os.path.join(args.results_dir, "raw_outputs.jsonl")
        metrics_file = os.path.join(args.results_dir, "metrics.json")
        details_file = os.path.join(args.results_dir, "chair_details.json") if args.save_details else None
    else:
        raise ValueError("Must specify either --results_dir or --results_file")

    evaluate_chair(
        results_file=results_file,
        cache_path=args.cache_path,
        coco_path=args.coco_path,
        output_metrics_file=metrics_file,
        save_details_file=details_file
    )

if __name__ == "__main__":
    main()
