#!/usr/bin/env python3
"""
Convert dumped generations from verl validation to EM/F1 evaluation format.

Reads the JSONL file produced by `_dump_generations()` (which has fields:
  input, output, gts, score, step, ...)
and converts to a format compatible with eval_em_f1.py, then computes EM/F1.

Usage:
    python convert_and_eval.py --dumped_jsonl path/to/dumped.jsonl
"""

import argparse
import json
import re
import sys
import os

# Add parent dir to path for eval_em_f1 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_em_f1 import evaluate_file, extract_answer_from_text


def extract_question_from_input(input_text: str) -> str:
    """Extract the question from the full prompt text."""
    # Try to find "Question:" or the last user content
    q_match = re.search(r'Question:\s*(.*?)(?:\n|$)', input_text)
    if q_match:
        return q_match.group(1).strip()
    # Fallback: return last non-empty line
    lines = [l.strip() for l in input_text.split('\n') if l.strip()]
    return lines[-1] if lines else input_text[:200]


def extract_ground_truth(gts_entry) -> list[str]:
    """Extract ground truth answers from the gts field (varies in structure)."""
    if gts_entry is None:
        return []
    if isinstance(gts_entry, dict):
        # Format: {"target": ["answer1", "answer2"]}
        targets = gts_entry.get("target", [])
        if isinstance(targets, list):
            return [str(t) for t in targets]
        return [str(targets)]
    if isinstance(gts_entry, list):
        return [str(t) for t in gts_entry]
    return [str(gts_entry)]


def convert_and_evaluate(dumped_jsonl: str, output_jsonl: str = None, verbose: bool = False):
    """
    Convert dumped generations and compute EM/F1.

    Args:
        dumped_jsonl: Path to dumped generations JSONL from _dump_generations()
        output_jsonl: Path to save converted JSONL (optional)
        verbose: Print per-sample details

    Returns:
        dict with EM/F1 results
    """
    # Read dumped JSONL and convert
    converted_lines = []
    total = 0
    parsed = 0

    with open(dumped_jsonl, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            example = json.loads(line)

            # Extract prediction from <answer> tags
            output_text = example.get("output", "")
            prediction = extract_answer_from_text(output_text)

            # Extract ground truth
            gts = example.get("gts", None)
            answers = extract_ground_truth(gts)

            # Only count if we have both prediction and ground truth
            if prediction or answers:
                parsed += 1

            question = extract_question_from_input(example.get("input", ""))

            converted = {
                "question": question,
                "output": output_text,
                "prediction": prediction,
                "answer": answers,
                "score": example.get("score", 0),
            }
            converted_lines.append(converted)

    if verbose:
        print(f"Read {total} lines from {dumped_jsonl}, parsed {parsed} with predictions")

    # Write converted JSONL
    if output_jsonl:
        os.makedirs(os.path.dirname(output_jsonl) or '.', exist_ok=True)
        with open(output_jsonl, 'w') as f:
            for item in converted_lines:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        if verbose:
            print(f"Converted data saved to {output_jsonl}")

    # Compute EM/F1 using the converted file as both predictions and ground truth
    # (each line has both 'output' and 'answer' fields)
    temp_file = output_jsonl or '/tmp/_eval_temp.jsonl'
    if not output_jsonl:
        with open(temp_file, 'w') as f:
            for item in converted_lines:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    results = evaluate_file(temp_file, temp_file)

    # Clean up temp file
    if not output_jsonl and os.path.exists(temp_file):
        os.remove(temp_file)

    return results


def main():
    parser = argparse.ArgumentParser(description="Convert and evaluate dumped generations")
    parser.add_argument("--dumped_jsonl", type=str, required=True,
                        help="Path to dumped generations JSONL from verl validation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for converted JSONL (optional)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample details")
    args = parser.parse_args()

    results = convert_and_evaluate(
        dumped_jsonl=args.dumped_jsonl,
        output_jsonl=args.output,
        verbose=args.verbose,
    )

    print(f"\n{'='*50}")
    print(f"Evaluation Results")
    print(f"{'='*50}")
    print(f"Total questions: {results['total']}")
    print(f"Exact Match (EM): {results['em']:.2f}%")
    print(f"F1 Score:         {results['f1']:.2f}%")
    print(f"Correct answers:  {results['correct']} / {results['total']}")
    print(f"{'='*50}\n")

    # Print per-dataset breakdown if we have data_source info
    if args.verbose and 'details' in results:
        from collections import Counter
        ds_counter = Counter()
        for d in results['details']:
            ds_counter['all'] += 1
        print(f"Samples by dataset: {dict(ds_counter)}")


if __name__ == "__main__":
    main()
