#!/usr/bin/env python3
"""
Generate teacher trajectories for SFT cold-start using a teacher model.

This script loads a teacher model (e.g., Search-R1 trained model or instruct model),
generates search trajectories on NQ/HotpotQA data, and saves them in a format
suitable for later Hermes conversion and SFT data building.

Key features:
    - Uses Qwen chat template for proper message formatting
    - Correctly extracts system prompt and user question from verl prompt format
    - Supports <search> query </search> or <tool_call>...</tool_call> generation
    - Calls local retrieval service for search results
    - Saves trajectories with metadata for filtering

Usage:
    python generate_teacher_trajectories.py \\
        --model_path /path/to/teacher_model \\
        --parquet_path /path/to/train.parquet \\
        --output_dir /path/to/output \\
        --nq_samples 10000 --hotpotqa_samples 10000 \\
        --retrieval_url http://127.0.0.1:8000/retrieve
"""

import argparse
import json
import os
import re
import time
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

# Note: torch and transformers are imported lazily in functions that need them
# to allow unit tests of utility functions without GPU dependencies.

# ---------------------------------------------------------------------------
# Prompt extraction from verl parquet format
# ---------------------------------------------------------------------------


def extract_prompt_messages(row: dict) -> tuple[str, str]:
    """Extract system prompt and user question from verl parquet row.

    The prompt field is a list of message dicts with 'role' and 'content'.
    Returns (system_prompt, question). Either may be empty string if not found.
    """
    prompt = row["prompt"]

    # Handle numpy/pandas array wrapping
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()

    system_prompt = ""
    question = ""

    for message in prompt:
        role = message.get("role")
        content = message.get("content", "")

        if role == "system":
            system_prompt = content
        elif role == "user":
            question = content

    if not question:
        raise ValueError("No user question found in prompt.")

    return system_prompt, question


# ---------------------------------------------------------------------------
# Search query extraction from model output
# ---------------------------------------------------------------------------

SEARCH_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def extract_search_query(text: str) -> Optional[str]:
    """Extract the last search query from model output.

    Supports both <search>query</search> and
    <tool_call>{"name":"search","arguments":{"query":"..."}}</tool_call> formats.
    Returns None if no search query found.
    """
    # Try Hermes <tool_call> format first
    tool_calls = TOOL_CALL_PATTERN.findall(text)
    if tool_calls:
        for raw_call in reversed(tool_calls):
            try:
                payload = json.loads(raw_call.strip())
                if (
                    isinstance(payload, dict)
                    and payload.get("name") == "search"
                ):
                    args = payload.get("arguments", {})
                    query = args.get("query", "")
                    if isinstance(query, str) and query.strip():
                        return query.strip()
            except (json.JSONDecodeError, TypeError):
                continue

    # Fall back to <search> format
    searches = SEARCH_PATTERN.findall(text)
    if searches:
        return searches[-1].strip()

    return None


# ---------------------------------------------------------------------------
# Retrieval service call
# ---------------------------------------------------------------------------


def call_retrieval(query: str, retrieval_url: str, topk: int = 3) -> str:
    """Call the local retrieval service and format results."""
    payload = {
        "queries": [query],
        "topk": topk,
        "return_scores": True,
    }
    resp = requests.post(retrieval_url, json=payload, timeout=10)
    resp.raise_for_status()
    results = resp.json()["result"]
    docs = results[0] if results else []

    parts = []
    for idx, doc_item in enumerate(docs):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        parts.append(f"Doc {idx + 1}(Title: {title}) {text}")

    if not parts:
        return "<information>No results found.</information>"

    return "<information>" + "\n\n".join(parts) + "</information>"


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------


def generate_trajectory(
    question: str,
    system_prompt: str,
    model,
    tokenizer,
    retrieval_url: str,
    device: str,
    max_turns: int = 3,
    topk: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[dict]:
    """Generate a full trajectory for one question using the teacher model.

    Uses HuggingFace transformers (single GPU). For multi-GPU VLLM, use
    generate_trajectory_vllm() instead.

    Uses Qwen chat template for proper message formatting at each turn.
    """
    import torch  # lazy import for GPU-dependent code

    messages = []

    # Add system prompt if available
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add user question
    messages.append({"role": "user", "content": question})

    for turn in range(max_turns):
        # Apply chat template
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        # Generate (ensure model_inputs is a dict, not a Tensor)
        with torch.no_grad():
            if hasattr(model_inputs, 'keys'):
                gen_kwargs = {k: v for k, v in model_inputs.items()}
            else:
                gen_kwargs = {'input_ids': model_inputs, 'attention_mask': torch.ones_like(model_inputs)}
            outputs = model.generate(
                **gen_kwargs,
                max_new_tokens=256,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=[
                    tokenizer.eos_token_id or tokenizer.pad_token_id,
                ],
            )

        # Decode only the new tokens
        input_len = model_inputs["input_ids"].shape[1]
        new_tokens = outputs[0, input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = response.strip()

        if not response:
            break

        messages.append({"role": "assistant", "content": response})

        # Check for search query
        query = extract_search_query(response)
        if query:
            try:
                search_results = call_retrieval(query, retrieval_url, topk=topk)
                messages.append({"role": "tool", "content": search_results})
            except Exception as e:
                error_msg = f"<information>Search failed: {e}</information>"
                messages.append({"role": "tool", "content": error_msg})
        else:
            # No search query — model is answering directly
            break

        # Check if answer is already given
        if "<answer>" in response and "</answer>" in response:
            break

    return messages


def generate_trajectory_vllm(
    question: str,
    system_prompt: str,
    llm,
    tokenizer,
    retrieval_url: str,
    max_turns: int = 3,
    topk: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[dict]:
    """Generate a full trajectory using VLLM for fast multi-GPU inference.

    Uses vllm.LLM.chat() with proper message format, no manual tokenization needed.
    4× L20 GPUs with tensor parallelism provides ~5-10x speedup over single GPU.
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=256,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )

    messages = []

    # Add system prompt if available
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add user question
    messages.append({"role": "user", "content": question})

    for turn in range(max_turns):
        # Generate with VLLM (handles tokenization internally)
        outputs = llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        response = outputs[0].outputs[0].text.strip()

        if not response:
            break

        messages.append({"role": "assistant", "content": response})

        # Check for search query
        query = extract_search_query(response)
        if query:
            try:
                search_results = call_retrieval(query, retrieval_url, topk=topk)
                messages.append({"role": "tool", "content": search_results})
            except Exception as e:
                error_msg = f"<information>Search failed: {e}</information>"
                messages.append({"role": "tool", "content": error_msg})
        else:
            # No search query — model is answering directly
            break

        # Check if answer is already given
        if "<answer>" in response and "</answer>" in response:
            break

    return messages


def main():
    parser = argparse.ArgumentParser(
        description="Generate teacher trajectories for SFT cold-start"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to teacher model")
    parser.add_argument("--parquet_path", type=str, required=True,
                        help="Path to input parquet file with NQ + HotpotQA data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for generated trajectories")
    parser.add_argument("--nq_samples", type=int, default=10000,
                        help="Number of NQ samples to process")
    parser.add_argument("--hotpotqa_samples", type=int, default=10000,
                        help="Number of HotpotQA samples to process")
    parser.add_argument("--retrieval_url", type=str,
                        default="http://127.0.0.1:8000/retrieve",
                        help="Retrieval service URL")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for generation")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (only batch_size=1 supported)")
    parser.add_argument("--max_turns", type=int, default=3,
                        help="Maximum assistant turns per trajectory")
    parser.add_argument("--topk", type=int, default=3,
                        help="Default topk for retrieval")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature (lower = more deterministic, use 0.3 for more search)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use VLLM for fast multi-GPU inference (requires vllm installed)")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="Number of GPUs for VLLM tensor parallelism (default: 4 for 4×L20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load model and tokenizer
    # -----------------------------------------------------------------------
    # Lazy imports for torch-heavy packages (allows testing utility functions
    # without GPU dependencies)
    import torch  # noqa: F401
    import transformers  # noqa: F401

    print(f"Loading model from {args.model_path} ...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_vllm:
        from vllm import LLM as VLLM
        print(f"Using VLLM with tensor_parallel_size={args.tensor_parallel_size} ...")
        llm = VLLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype="bfloat16",
            trust_remote_code=True,
            seed=args.seed,
        )
        model = None
        device = None
        use_vllm = True
        print(f"VLLM model loaded. Using {args.tensor_parallel_size} GPUs.")
    else:
        print("Loading with HuggingFace Transformers (single GPU)...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = model.to(dtype=torch.bfloat16)
        model.eval()
        print(f"Model loaded. Device: {model.device}")
        llm = None
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        use_vllm = False

    # -----------------------------------------------------------------------
    # Load and sample data
    # -----------------------------------------------------------------------
    print(f"Loading data from {args.parquet_path} ...")
    df = pd.read_parquet(args.parquet_path)

    nq_df = df[df["data_source"] == "nq"].sample(
        n=min(args.nq_samples, len(df[df["data_source"] == "nq"])),
        random_state=args.seed,
    )
    hp_df = df[df["data_source"] == "hotpotqa"].sample(
        n=min(args.hotpotqa_samples, len(df[df["data_source"] == "hotpotqa"])),
        random_state=args.seed,
    )

    subset = pd.concat([nq_df, hp_df])
    subset = subset.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"Total samples: {len(subset)} (NQ: {len(nq_df)}, HotpotQA: {len(hp_df)})")

    # -----------------------------------------------------------------------
    # Generate trajectories
    # -----------------------------------------------------------------------
    output_file = os.path.join(args.output_dir, "teacher_trajectories.jsonl")
    stats = {
        "total": 0,
        "has_search": 0,
        "has_answer": 0,
        "has_tool_call": 0,
        "errors": 0,
    }

    if not args.use_vllm:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(output_file, "w") as fout:
        for _, row in tqdm(subset.iterrows(), total=len(subset)):
            # Extract system prompt and question from verl prompt format
            try:
                system_prompt, question = extract_prompt_messages(row)
            except (ValueError, KeyError) as e:
                print(f"  Skipping row: {e}")
                stats["errors"] += 1
                continue

            targets_raw = row.get("reward_model", {}).get("ground_truth", {}).get("target", [])
            # Handle numpy arrays and other non-JSON-serializable types
            if hasattr(targets_raw, "tolist"):
                targets = targets_raw.tolist()
            elif isinstance(targets_raw, (list, tuple)):
                targets = list(targets_raw)
            else:
                targets = [str(targets_raw)]
            # Ensure all elements are plain Python types
            targets = [str(t) if not isinstance(t, (str, int, float)) else t for t in targets]
            data_source = row.get("data_source", "unknown")

            # Generate trajectory
            try:
                if args.use_vllm:
                    trajectory = generate_trajectory_vllm(
                        question=question,
                        system_prompt=system_prompt,
                        llm=llm,
                        tokenizer=tokenizer,
                        retrieval_url=args.retrieval_url,
                        max_turns=args.max_turns,
                        topk=args.topk,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                else:
                    trajectory = generate_trajectory(
                        question=question,
                        system_prompt=system_prompt,
                        model=model,
                        tokenizer=tokenizer,
                        retrieval_url=args.retrieval_url,
                        device=device,
                        max_turns=args.max_turns,
                        topk=args.topk,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
            except Exception as e:
                import traceback
                print(f"  Error generating trajectory for '{question[:50]}...': {e}")
                traceback.print_exc()
                stats["errors"] += 1
                continue

            # Analyze trajectory
            has_search_tag = any(
                "<search>" in t.get("content", "") for t in trajectory
            )
            has_tool_call_tag = any(
                "<tool_call>" in t.get("content", "") for t in trajectory
            )
            has_answer_tag = any(
                "<answer>" in t.get("content", "") for t in trajectory
            )

            num_search = sum(
                1 for t in trajectory
                if "<search>" in t.get("content", "") or "<tool_call>" in t.get("content", "")
            )

            stats["total"] += 1
            if has_search_tag or has_tool_call_tag:
                stats["has_search"] += 1
            if has_tool_call_tag:
                stats["has_tool_call"] += 1
            if has_answer_tag:
                stats["has_answer"] += 1

            record = {
                "question": question,
                "system_prompt": system_prompt,
                "trajectory": trajectory,
                "answer": targets,
                "data_source": data_source,
                "retrieval_success": has_search_tag or has_tool_call_tag,
                "trajectory_valid": (has_search_tag or has_tool_call_tag) and has_answer_tag,
                "num_tool_calls": num_search,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\nDone! Generated {stats['total']} trajectories.")
    print(f"  Has search:  {stats['has_search']} ({stats['has_search'] / max(stats['total'], 1) * 100:.1f}%)")
    print(f"  Has tool_call: {stats['has_tool_call']} ({stats['has_tool_call'] / max(stats['total'], 1) * 100:.1f}%)")
    print(f"  Has answer:  {stats['has_answer']} ({stats['has_answer'] / max(stats['total'], 1) * 100:.1f}%)")
    print(f"  Errors:      {stats['errors']}")
    print(f"  Output:      {output_file}")


if __name__ == "__main__":
    main()
