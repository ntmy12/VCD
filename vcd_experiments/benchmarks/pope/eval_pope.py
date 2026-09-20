import os
import json
import argparse
from typing import Dict, Any, List

def parse_pred(pred_text: str) -> str:
    """
    Robust rule-based parser to extract 'yes' or 'no' from model generated text.
    Handles punctuation, capitalization, and multi-word responses.
    """
    if not pred_text:
        return "unknown"
    text = str(pred_text).strip()
    
    # Strip chat template / conversation prefix if present (e.g., ASSISTANT:)
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    elif "assistant\n" in text.lower():
        idx = text.lower().rfind("assistant\n")
        text = text[idx + len("assistant\n"):].strip()
        
    text = text.lower()
    
    # Strip leading/trailing punctuation
    cleaned = text.strip(" .,!?:;\n\t\"'()[]{}")
    words = cleaned.split()
    if not words:
        return "unknown"
        
    first_word = words[0].strip(" .,!?:;\n\t\"'()[]{}")
    if first_word in ["yes", "yeah", "yup", "y"]:
        return "yes"
    elif first_word in ["no", "nope", "nah", "n"]:
        return "no"
        
    # Check within first 5 words
    first_5 = [w.strip(" .,!?:;\n\t\"'()[]{}") for w in words[:5]]
    has_yes = "yes" in first_5
    has_no = "no" in first_5
    
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    if has_yes and has_no:
        return "yes" if first_5.index("yes") < first_5.index("no") else "no"
        
    return "unknown"

def evaluate_predictions(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes POPE metrics: Accuracy, Precision, Recall, F1-Score, Yes-ratio, Unknowns.
    Compatible with keys 'pred', 'text', 'answer' for predictions, and 'label', 'gt_answer' for ground truth.
    """
    TP = TN = FP = FN = 0
    unknown = 0
    yes_answers = 0
    total = len(records)
    
    if total == 0:
        return {
            "Accuracy": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "Yes_ratio": 0.0,
            "Total": 0,
            "TP": 0, "TN": 0, "FP": 0, "FN": 0, "Unknown": 0
        }
        
    for item in records:
        gt_label = item.get("label", item.get("gt_answer", "")).lower().strip()
        raw_pred = item.get("pred", item.get("text", item.get("answer", "")))
        pred = parse_pred(raw_pred)
        
        if pred == "yes":
            yes_answers += 1
            if gt_label == "yes":
                TP += 1
            else:
                FP += 1
        elif pred == "no":
            if gt_label == "no":
                TN += 1
            else:
                FN += 1
        else:
            unknown += 1
            # Unknown treated as incorrect prediction
            if gt_label == "yes":
                FN += 1
            else:
                FP += 1

    accuracy = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    yes_ratio = yes_answers / total if total > 0 else 0.0
    
    metrics = {
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1": round(f1 * 100, 2),
        "Yes_ratio": round(yes_ratio * 100, 2),
        "Total": total,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Unknown": unknown
    }
    return metrics

def eval_pope_file(results_file: str) -> Dict[str, Any]:
    """
    Evaluates a single raw_outputs.jsonl file.
    """
    with open(results_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return evaluate_predictions(records)

def main():
    parser = argparse.ArgumentParser(description="Evaluate POPE Benchmark Results")
    parser.add_argument("--results_file", type=str, default=None, help="Path to raw_outputs.jsonl")
    parser.add_argument("--results_dir", type=str, default=None, help="Path to directory containing raw_outputs.jsonl")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save metrics.json")
    args = parser.parse_args()
    
    if args.results_file:
        file_path = args.results_file
        save_dir = os.path.dirname(file_path)
    elif args.results_dir:
        file_path = os.path.join(args.results_dir, "raw_outputs.jsonl")
        save_dir = args.results_dir
    else:
        raise ValueError("Must specify either --results_file or --results_dir")
        
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}")
        
    metrics = eval_pope_file(file_path)
    
    print("=" * 45)
    print(f" POPE Evaluation Results: {os.path.basename(file_path)}")
    print("=" * 45)
    print(f"  Accuracy:    {metrics['Accuracy']}%")
    print(f"  Precision:   {metrics['Precision']}%")
    print(f"  Recall:      {metrics['Recall']}%")
    print(f"  F1-Score:    {metrics['F1']}%")
    print(f"  Yes Ratio:   {metrics['Yes_ratio']}%")
    print(f"  Total:       {metrics['Total']} samples")
    print(f"  Unknown:     {metrics['Unknown']}")
    print("=" * 45)
    
    out_path = args.output_file or os.path.join(save_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to: {out_path}")

if __name__ == "__main__":
    main()
