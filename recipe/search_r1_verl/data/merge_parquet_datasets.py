#!/usr/bin/env python3
"""
Merge NQ and HotpotQA parquet files into a single dataset.

This script solves the file overwrite issue by first converting each dataset
to its own file (nq_train.parquet, hotpotqa_train.parquet), then merging
them into a single train.parquet for use in training.

Usage:
    # Step 1: Convert each dataset separately
    python convert_nq_to_parquet.py --output_dir /path/to/data --split train
    python convert_hotpotqa_to_parquet.py --output_dir /path/to/data --split train

    # Step 2: Merge
    python merge_parquet_datasets.py \\
        --input_dir /path/to/data \\
        --split train

    # Or process all splits
    python merge_parquet_datasets.py \\
        --input_dir /path/to/data \\
        --split all
"""

import argparse
import os

import pandas as pd


def merge_split(input_dir: str, split: str):
    """Merge NQ and HotpotQA parquet files for a given split."""
    nq_path = os.path.join(input_dir, f"nq_{split}.parquet")
    hotpotqa_path = os.path.join(input_dir, f"hotpotqa_{split}.parquet")
    output_path = os.path.join(input_dir, f"{split}.parquet")

    parts = []

    if os.path.exists(nq_path):
        nq_df = pd.read_parquet(nq_path)
        print(f"NQ {split}: {len(nq_df)} records")
        parts.append(nq_df)
    else:
        print(f"Warning: {nq_path} not found, skipping NQ")

    if os.path.exists(hotpotqa_path):
        hotpotqa_df = pd.read_parquet(hotpotqa_path)
        print(f"HotpotQA {split}: {len(hotpotqa_df)} records")
        parts.append(hotpotqa_df)
    else:
        print(f"Warning: {hotpotqa_path} not found, skipping HotpotQA")

    if not parts:
        print(f"No data found for split '{split}', skipping.")
        return

    merged_df = pd.concat(parts, ignore_index=True)
    merged_df.to_parquet(output_path, index=False)
    print(f"Merged {len(merged_df)} records -> {output_path}")
    print(f"  Data sources: {merged_df['data_source'].value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge NQ and HotpotQA parquet files"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing nq_*.parquet and hotpotqa_*.parquet")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test", "all"],
                        help="Split to merge, or 'all' to merge all splits")
    args = parser.parse_args()

    if args.split == "all":
        for split in ["train", "validation", "test"]:
            merge_split(args.input_dir, split)
    else:
        merge_split(args.input_dir, args.split)


if __name__ == "__main__":
    main()
