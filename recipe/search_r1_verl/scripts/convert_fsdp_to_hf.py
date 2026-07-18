#!/usr/bin/env python3
"""
Convert FSDP2 checkpoint to HuggingFace format.

Usage:
    python convert_fsdp_to_hf.py \
        --fsdp_ckpt_dir /path/to/global_step_300/actor \
        --output_dir /path/to/hf_model
"""

import argparse
import os
import torch

# Must import DTensor before loading checkpoint
import torch.distributed.tensor as dtensor


def convert_fsdp_to_hf(fsdp_ckpt_dir: str, output_dir: str, hf_base_model: str = None):
    """
    Convert FSDP2 checkpoint shards to HuggingFace format.

    Args:
        fsdp_ckpt_dir: Path to FSDP checkpoint directory (containing model_world_size_*_rank_*.pt)
        output_dir: Output path for HuggingFace model
        hf_base_model: Path to base HF model (for config, tokenizer). If None, uses huggingface/ subdir.
    """
    import json
    from collections import OrderedDict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Determine HF base path
    hf_config_path = hf_base_model
    if hf_config_path is None:
        hf_config_path = os.path.join(fsdp_ckpt_dir, "huggingface")
    
    if not os.path.exists(hf_config_path):
        raise FileNotFoundError(
            f"HF config path not found: {hf_config_path}. "
            "Please provide --hf_base_model pointing to the original SFT model."
        )

    print(f"Using HF config/tokenizer from: {hf_config_path}")
    print(f"Loading FSDP shards from: {fsdp_ckpt_dir}")

    # Find all model shards
    shard_files = sorted([
        f for f in os.listdir(fsdp_ckpt_dir)
        if f.startswith("model_world_size") and f.endswith(".pt")
    ])
    
    if not shard_files:
        raise FileNotFoundError(f"No model_world_size_*_rank_*.pt files found in {fsdp_ckpt_dir}")
    
    print(f"Found {len(shard_files)} shard(s): {shard_files}")

    # Load and merge all shards
    full_state_dict = OrderedDict()

    for shard_file in shard_files:
        shard_path = os.path.join(fsdp_ckpt_dir, shard_file)
        print(f"  Loading {shard_file}...")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        
        for key, param in shard.items():
            # DTensor needs to be gathered to full tensor
            if isinstance(param, dtensor.DTensor):
                param = param.to_local()  # Get local shard
                # Note: For FSDP2, to_local() gives the shard on this "rank".
                # Since we loaded from a specific rank's file, this is correct.
            
            # Clean up FSDP-specific key prefixes
            clean_key = key.replace("_fsdp_wrapped_module.", "")
            
            if clean_key in full_state_dict:
                print(f"  WARNING: Duplicate key {clean_key}")
            
            if isinstance(param, torch.Tensor):
                full_state_dict[clean_key] = param.contiguous().to(dtype=torch.bfloat16)
            else:
                print(f"  Skipping non-tensor value for {clean_key}: {type(param)}")

    print(f"Merged state dict: {len(full_state_dict)} keys")

    # Load model architecture
    print(f"Loading model config from {hf_config_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_config_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    
    # Get expected keys
    expected_keys = set(model.state_dict().keys())
    provided_keys = set(full_state_dict.keys())
    
    missing_keys = expected_keys - provided_keys
    extra_keys = provided_keys - expected_keys
    
    if missing_keys:
        print(f"WARNING: {len(missing_keys)} missing keys (will be left as initialized):")
        for k in sorted(list(missing_keys))[:10]:
            print(f"  - {k}")
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys) - 10} more")
    
    if extra_keys:
        print(f"WARNING: {len(extra_keys)} extra keys (will be ignored):")
        for k in sorted(list(extra_keys))[:5]:
            print(f"  - {k}")
        if len(extra_keys) > 5:
            print(f"  ... and {len(extra_keys) - 5} more")
    
    # Load weights
    model.load_state_dict(full_state_dict, strict=False)
    
    # Save to output
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving HF model to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    
    # Save tokenizer
    print(f"Saving tokenizer to {output_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_config_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Conversion complete! Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert FSDP2 checkpoint to HuggingFace format")
    parser.add_argument("--fsdp_ckpt_dir", type=str, required=True,
                        help="Path to FSDP checkpoint directory (e.g., global_step_300/actor)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output path for HuggingFace model")
    parser.add_argument("--hf_base_model", type=str, default=None,
                        help="Path to base HF model (for config/tokenizer). Default: uses huggingface/ subdir")
    args = parser.parse_args()

    convert_fsdp_to_hf(
        fsdp_ckpt_dir=args.fsdp_ckpt_dir,
        output_dir=args.output_dir,
        hf_base_model=args.hf_base_model,
    )


if __name__ == "__main__":
    main()
