#!/usr/bin/env python3
"""
Convert HotpotQA dataset to verl-compatible parquet format.

Output format (raw chat):
    {
        "data_source": "hotpotqa",
        "prompt": [{"role": "user", "content": "Question ..."}],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": ["answer1", "answer2"]}
        },
        "extra_info": {"split": "train", "index": 0, "type": "bridge"}
    }

Usage:
    python convert_hotpotqa_to_parquet.py \\
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


def extract_hotpotqa_answer(example: dict) -> list[str]:
    """Extract answer from HotpotQA example."""
    answer = example.get("answer", "").strip()
    if answer:
        return [answer]
    return []


def main():
    parser = argparse.ArgumentParser(description="Convert HotpotQA to verl parquet")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for parquet files")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test"],
                        help="Dataset split to process")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load HotpotQA dataset
    print(f"Loading HotpotQA {args.split} split...")
    # Use the 'distractor' config which is the standard HotpotQA setting
    dataset = datasets.load_dataset(
        "hotpot_qa",
        "distractor",
        split=args.split,
    )
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Processing {len(dataset)} samples...")

    records = []
    skipped = 0

    for idx, example in enumerate(dataset):
        question = example.get("question", "").strip()
        answers = extract_hotpotqa_answer(example)
        q_type = example.get("type", "")

        if not question:
            skipped += 1
            continue

        if not answers:
            skipped += 1
            continue

        record = {
            "data_source": "hotpotqa",
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
                "type": q_type,
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
