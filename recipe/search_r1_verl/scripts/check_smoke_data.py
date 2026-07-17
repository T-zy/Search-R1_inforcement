#!/usr/bin/env python3
"""检查 smoke test 数据的 prompt 长度"""
import pandas as pd
from transformers import AutoTokenizer

SFT_PATH = "/media/public/RAIDStorageArray/workdir/zytan/checkpoints/qwen2.5-1.5b-searchr1-sft-merged"
df = pd.read_parquet("/home/zytan/Search-R1_inforcement/data/smoke_test/train.parquet")
tok = AutoTokenizer.from_pretrained(SFT_PATH, trust_remote_code=True)

for i, row in df.iterrows():
    prompt = row["prompt"]
    text = ""
    for msg in prompt:
        text += str(msg.get("content", ""))
    tokens = tok.encode(text)
    source = row["data_source"]
    print(f"Sample {i}: {len(tokens)} tokens, source={source}")
