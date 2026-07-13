#!/usr/bin/env python3
"""
Convert Natural Questions (NQ) dataset to verl-compatible parquet format.

Output format (raw chat):
    {
        "data_source": "nq",
        "prompt": [{"role": "user", "content": "Question ..."}],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": ["answer1", "answer2"]}
        },
        "extra_info": {"split": "train", "index": 0}
    }

Usage:
    python convert_nq_to_parquet.py \\
        --output_dir /path/to/output \\
        --split train
"""

import argparse
import json
import os
import re

import datasets
import pandas as pd


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    def remove_punc(text: str) -> str:
        exclude = set(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text: str) -> str:
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_nq_answer(example: dict) -> list[str]:
    """Extract short answers from NQ example."""
    answers = set()
    if "annotations" in example and example["annotations"]:
        for annotation in example["annotations"]:
            if "short_answers" in annotation and annotation["short_answers"]:
                for sa in annotation["short_answers"]:
                    if sa and "text" in sa and sa["text"]:
                        for t in sa["text"]:
                            if t.strip():
                                answers.add(t.strip())
            # Also check yes/no answers
            if "yes_no_answer" in annotation:
                yn = annotation["yes_no_answer"]
                if yn in ("YES", "NO"):
                    answers.add(yn.lower())
    return list(answers) if answers else []


def extract_nq_question(example: dict) -> str:
    """Extract question text from NQ example."""
    return example.get("question", {}).get("text", "").strip()


def main():
    parser = argparse.ArgumentParser(description="Convert NQ to verl parquet")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for parquet files")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test"],
                        help="Dataset split to process")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load NQ dataset
    print(f"Loading NQ {args.split} split...")
    dataset = datasets.load_dataset(
        "nq_open",
        split=args.split,
    )
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Processing {len(dataset)} samples...")

    records = []
    skipped = 0

    for idx, example in enumerate(dataset):
        question = extract_nq_question(example)
        answers = extract_nq_answer(example)

        if not question:
            skipped += 1
            continue

        # For nq_open, answers are directly available
        if "answer" in example:
            answers_text = example["answer"]
            if isinstance(answers_text, list):
                answers = [a.strip() for a in answers_text if a.strip()]
            elif isinstance(answers_text, str):
                answers = [answers_text.strip()]

        if not answers:
            skipped += 1
            continue

        record = {
            "data_source": "nq",
            "prompt": [
                {"role": "user", "content": question}
            ],
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "target": answers
                }
            },
            "extra_info": {
                "split": args.split,
                "index": idx,
            }
        }
        records.append(record)

    print(f"Processed: {len(records)} valid, {skipped} skipped")

    # Save as parquet
    output_path = os.path.join(args.output_dir, f"{args.split}.parquet")
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
