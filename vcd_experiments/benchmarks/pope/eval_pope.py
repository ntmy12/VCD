import argparse
import json
import os
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate POPE results")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the directory containing raw_outputs.jsonl")
    return parser.parse_args()

def parse_answer(text):
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Common matching rules
    if text.startswith('yes'):
        return 'yes'
    if text.startswith('no'):
        return 'no'
    
    # Fallback to check if yes/no is in the string at all
    if 'yes' in text and 'no' not in text:
        return 'yes'
    if 'no' in text and 'yes' not in text:
        return 'no'
        
    return 'unknown'

def main():
    args = parse_args()
    
    raw_outputs_file = os.path.join(args.results_dir, "raw_outputs.jsonl")
    if not os.path.exists(raw_outputs_file):
        print(f"Error: Could not find {raw_outputs_file}")
        return
        
    y_true = []
    y_pred = []
    
    # Parse inputs
    with open(raw_outputs_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            gt = item['ground_truth'].lower().strip()
            pred = parse_answer(item['generated_text'])
            
            y_true.append(1 if gt == 'yes' else 0)
            
            if pred == 'yes':
                y_pred.append(1)
            elif pred == 'no':
                y_pred.append(0)
            else:
                # If unknown, we consider it a wrong prediction
                # If ground truth is yes, we predict no; if no, we predict yes
                y_pred.append(0 if gt == 'yes' else 1)

    # Compute metrics
    TP = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1])
    TN = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0])
    FP = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1])
    FN = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0])

    accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "samples_evaluated": len(y_true)
    }
    
    metrics_file = os.path.join(args.results_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\n--- POPE EVALUATION RESULTS ---")
    print(f"Directory: {args.results_dir}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()
