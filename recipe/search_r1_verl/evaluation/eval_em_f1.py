#!/usr/bin/env python3
"""
EM and F1 evaluation for Search-R1 model outputs.

Computes Exact Match (EM) and F1 score between predicted answers
and ground truth answers, matching the original Search-R1 evaluation protocol.

Usage:
    python eval_em_f1.py --predictions preds.jsonl --ground_truth gtruth.jsonl
"""

import argparse
import json
import re
import string
from collections import Counter
from typing import Any


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text: str) -> str:
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / max(len(pred_tokens), 1)
    recall = num_same / max(len(gt_tokens), 1)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def extract_answer_from_text(text: str) -> str:
    """Extract the last <answer>...</answer> block from model output."""
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, text, re.DOTALL))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def evaluate_file(predictions_file: str, ground_truth_file: str) -> dict[str, Any]:
    """
    Evaluate predictions against ground truth.

    Supports two input formats:
    1. JSONL with 'question', 'prediction', 'answer' keys
    2. JSONL with 'question', 'trajectory' (list of messages), 'answer' keys

    Returns dict with EM, F1, and per-question details.
    """
    # Load ground truth
    gt_data = {}
    with open(ground_truth_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            q = example.get("question", "")
            answers = example.get("answer", [])
            if isinstance(answers, str):
                answers = [answers]
            gt_data[q] = answers

    # Load predictions
    em_scores = []
    f1_scores = []
    details = []

    with open(predictions_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            question = example.get("question", "")

            # Extract prediction
            if "prediction" in example:
                prediction = example["prediction"]
            elif "trajectory" in example:
                # Extract from trajectory
                traj = example["trajectory"]
                prediction = ""
                for turn in reversed(traj):
                    if turn.get("role") == "assistant":
                        prediction = extract_answer_from_text(turn.get("content", ""))
                        if prediction:
                            break
            elif "output" in example:
                prediction = extract_answer_from_text(example["output"])
            else:
                prediction = ""

            # Get ground truth
            gt_answers = example.get("answer", []) if "answer" in example else gt_data.get(question, [])
            if isinstance(gt_answers, str):
                gt_answers = [gt_answers]

            if not prediction:
                em_scores.append(0)
                f1_scores.append(0.0)
                details.append({
                    "question": question,
                    "prediction": prediction,
                    "ground_truth": gt_answers,
                    "em": 0,
                    "f1": 0.0,
                })
                continue

            # Take best EM/F1 across multiple ground truth answers
            best_em = 0
            best_f1 = 0.0
            for gt in gt_answers:
                best_em = max(best_em, em_score(prediction, gt))
                best_f1 = max(best_f1, f1_score(prediction, gt))

            em_scores.append(best_em)
            f1_scores.append(best_f1)
            details.append({
                "question": question,
                "prediction": prediction,
                "ground_truth": gt_answers,
                "em": best_em,
                "f1": best_f1,
            })

    # Aggregate
    total = len(em_scores)
    avg_em = sum(em_scores) / max(total, 1) * 100
    avg_f1 = sum(f1_scores) / max(total, 1) * 100

    return {
        "total": total,
        "em": avg_em,
        "f1": avg_f1,
        "correct": sum(em_scores),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate EM and F1 for Search-R1")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSONL file")
    parser.add_argument("--ground_truth", type=str, required=True,
                        help="Path to ground truth JSONL file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for detailed results (optional)")
    args = parser.parse_args()

    results = evaluate_file(args.predictions, args.ground_truth)

    print(f"\n{'='*50}")
    print(f"Evaluation Results")
    print(f"{'='*50}")
    print(f"Total questions: {results['total']}")
    print(f"Exact Match (EM): {results['em']:.2f}%")
    print(f"F1 Score:         {results['f1']:.2f}%")
    print(f"Correct answers:  {results['correct']} / {results['total']}")
    print(f"{'='*50}\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
