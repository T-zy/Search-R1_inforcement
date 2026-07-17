#!/usr/bin/env python3
"""
Build SFT data from teacher trajectories for cold-start fine-tuning.

This script converts teacher model trajectories (containing CoT, search calls,
tool responses, and final answers) into training data for SFT cold-start.

Expected input format (JSONL) — preferably Hermes <tool_call> format:
    {
        "question": "...",
        "trajectory": [
            {"role": "user", "content": "Question ..."},
            {"role": "assistant", "content": "...<tool_call>...{...}...</tool_call>..."},
            {"role": "tool", "content": "<information>...</information>"},
            {"role": "assistant", "content": "...<answer>...</answer>..."}
        ],
        "answer": "golden_answer",
        "retrieval_success": true,
        "trajectory_valid": true,
        "num_tool_calls": 3
    }

Filtering rules:
    - Keep: EM correct, >=1 valid search, tool response not truncated,
      JSON-legal tool calls (Hermes format), has final answer, length within limit,
      valid role sequence, no failed retrievals
    - Discard: wrong answer, illegal format, retrieval failure,
      exceeds max turns, no final answer, too long, invalid role sequence

Usage:
    python build_sft_data.py \\
        --input /path/to/hermes_trajectories.jsonl \\
        --output /path/to/sft_data.jsonl \\
        --tokenizer_path /path/to/tokenizer \\
        --max_length 4096 \\
        --max_turns 3 \\
        --max_tool_calls 2
"""

import argparse
import json
import logging
import os
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool call pattern — Hermes <tool_call> format (JSON inside)
# ---------------------------------------------------------------------------
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

ANSWER_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.DOTALL,
)

SEARCH_FAILURE_MARKERS = (
    "Search failed:",
    "timed out",
    "HTTP error",
    "Unexpected error",
    "Query is empty",
)

# ---------------------------------------------------------------------------
# Tool schema for SFT data — matches GRPO stage
# ---------------------------------------------------------------------------
SEARCH_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search Wikipedia passages relevant "
                "to a factual question. Returns the "
                "top-k passages formatted as "
                "<information>...</information>."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query string."
                        )
                    },
                    "topk": {
                        "type": "integer",
                        "description": (
                            "Number of passages to "
                            "retrieve, from 1 to 5."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# Answer normalization and EM check
# ---------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
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


def em_check(prediction: str, golden_answers: list[str]) -> bool:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return True
    return False


# ---------------------------------------------------------------------------
# Hermes tool call parsing with strict validation
# ---------------------------------------------------------------------------


def parse_tool_calls(content: str) -> list[dict]:
    """Parse Hermes <tool_call> blocks with strict JSON validation.

    Returns a list of parsed tool call dicts (each has 'name' and 'arguments').

    Raises:
        ValueError: If any tool call has invalid JSON, wrong tool name,
                    missing/empty query, or invalid topk.
    """
    calls = []
    for raw_call in TOOL_CALL_PATTERN.findall(content):
        raw_stripped = raw_call.strip()
        if not raw_stripped:
            continue
        try:
            payload = json.loads(raw_stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool_call is not valid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError("Tool call payload must be a JSON object.")

        if payload.get("name") != "search":
            raise ValueError(
                f"Tool name must be 'search', got '{payload.get('name')}'."
            )

        arguments = payload.get("arguments")

        if not isinstance(arguments, dict):
            raise ValueError(
                "Tool arguments must be a JSON object."
            )

        query = arguments.get("query")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Search query must be a non-empty string."
            )

        topk = arguments.get("topk", 3)

        if not isinstance(topk, int):
            raise ValueError(
                f"topk must be an integer, got {type(topk).__name__}."
            )

        if not 1 <= topk <= 5:
            raise ValueError(
                f"topk must be between 1 and 5, got {topk}."
            )

        calls.append(payload)

    return calls


def validate_tool_calls(trajectory: list[dict]) -> tuple[bool, str]:
    """Validate all tool calls in a trajectory.

    Returns:
        (is_valid, reason) tuple. If valid, reason is empty string.
    """
    try:
        total_calls = 0
        for turn in trajectory:
            if turn.get("role") != "assistant":
                continue
            content = turn.get("content", "")
            total_calls += len(parse_tool_calls(content))

        if total_calls == 0:
            return False, "no_tool_call"

        return True, ""

    except ValueError as exc:
        return False, str(exc)


def count_tool_calls(trajectory: list[dict]) -> int:
    """Count the number of valid Hermes tool calls in the trajectory."""
    count = 0
    for turn in trajectory:
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content", "")
        try:
            count += len(parse_tool_calls(content))
        except ValueError:
            # Invalid calls are still counted for filtering purposes
            count += len(TOOL_CALL_PATTERN.findall(content))
    return count


# ---------------------------------------------------------------------------
# Role sequence validation
# ---------------------------------------------------------------------------


def validate_role_sequence(messages: list[dict]) -> tuple[bool, str]:
    """Validate the role alternation pattern in the trajectory.

    Rules:
        - All roles must be one of: system, user, assistant, tool
        - First non-system message must be user
        - A tool message must be preceded by an assistant message with a tool_call
        - An assistant tool_call must be followed by a tool message

    Returns:
        (is_valid, reason) tuple.
    """
    valid_roles = {"system", "user", "assistant", "tool"}

    for message in messages:
        role = message.get("role")
        if role not in valid_roles:
            return False, f"unknown_role: {role}"

    non_system = [
        m for m in messages if m.get("role") != "system"
    ]

    if not non_system:
        return False, "empty_messages"

    if non_system[0].get("role") != "user":
        return False, "missing_user"

    for index, message in enumerate(non_system):
        role = message.get("role")
        content = message.get("content", "")

        if role == "tool":
            if index == 0:
                return False, "orphan_tool_response"

            previous = non_system[index - 1]
            if previous.get("role") != "assistant":
                return False, "orphan_tool_response"

            if "<tool_call>" not in previous.get("content", ""):
                return False, "tool_without_tool_call"

        if role == "assistant":
            if "<tool_call>" not in content:
                continue

            if index + 1 >= len(non_system):
                return False, "missing_tool_response"

            next_message = non_system[index + 1]
            if next_message.get("role") != "tool":
                return False, "missing_tool_response"

    return True, ""


# ---------------------------------------------------------------------------
# Failed search detection
# ---------------------------------------------------------------------------


def has_failed_tool_response(trajectory: list[dict]) -> bool:
    """Check if any tool response contains search failure markers."""
    for turn in trajectory:
        if turn.get("role") != "tool":
            continue

        content = turn.get("content", "")

        if any(marker in content for marker in SEARCH_FAILURE_MARKERS):
            return True

        if "<information>" not in content or "</information>" not in content:
            return True

        inner = re.search(
            r"<information>(.*?)</information>",
            content,
            flags=re.DOTALL
        )

        if not inner or not inner.group(1).strip():
            return True

    return False


# ---------------------------------------------------------------------------
# Final answer extraction (strict)
# ---------------------------------------------------------------------------


def extract_final_answer(trajectory: list[dict]) -> str:
    """Extract the final answer from the last <answer>...</answer> tag."""
    for turn in reversed(trajectory):
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content", "")
        matches = ANSWER_PATTERN.findall(content)
        if matches:
            return matches[-1].strip()
    return ""


def has_final_answer(trajectory: list[dict]) -> bool:
    """Check if the trajectory has at least one complete <answer>...</answer> tag."""
    for turn in reversed(trajectory):
        if turn.get("role") == "assistant":
            content = turn.get("content", "")
            if ANSWER_PATTERN.search(content):
                return True
    return False


# ---------------------------------------------------------------------------
# Token length checking
# ---------------------------------------------------------------------------


def check_token_length(
    trajectory: list[dict],
    tokenizer,
    tools_schema: list[dict],
    max_length: int,
) -> bool:
    """Check if the total tokenized length (messages + tools) is within limit.

    Returns True if within limit, False otherwise.
    """
    if tokenizer is None:
        # If no tokenizer available, skip length check
        return True

    try:
        token_ids = tokenizer.apply_chat_template(
            trajectory,
            tools=tools_schema,
            tokenize=True,
            add_generation_prompt=False,
        )
        token_length = len(token_ids)
        return token_length <= max_length
    except Exception:
        # If tokenization fails, conservatively reject
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build SFT data from teacher trajectories")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with teacher trajectories (Hermes format)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file for SFT training")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer for length checking (required for --max_length to work)")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length in tokens (requires --tokenizer_path)")
    parser.add_argument("--max_turns", type=int, default=3,
                        help="Maximum number of assistant turns (should match GRPO max_assistant_turns)")
    parser.add_argument("--max_tool_calls", type=int, default=2,
                        help="Maximum number of tool calls (should match GRPO settings)")
    args = parser.parse_args()

    # Load tokenizer if path provided
    tokenizer = None
    if args.tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_path,
                trust_remote_code=True,
            )
            logger.info(f"Loaded tokenizer from {args.tokenizer_path}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {args.tokenizer_path}: {e}")
            logger.warning("Token length filtering will be disabled.")
            tokenizer = None
    else:
        if args.max_length != 4096:
            logger.warning(
                "--max_length is set but --tokenizer_path is not provided. "
                "Token length filtering will be disabled."
            )

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    stats = {
        "total": 0,
        "kept": 0,
        "no_tool_call": 0,
        "invalid_tool_json": 0,
        "invalid_tool_name": 0,
        "empty_query": 0,
        "failed_retrieval": 0,
        "missing_tool_response": 0,
        "invalid_role_sequence": 0,
        "missing_final_answer": 0,
        "wrong_answer": 0,
        "too_many_turns": 0,
        "too_many_tool_calls": 0,
        "too_long": 0,
    }

    # Per-dataset statistics
    dataset_stats: dict[str, dict] = {}

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)

            stats["total"] += 1

            question = example.get("question", "")
            trajectory = example.get("trajectory", [])
            golden_answers = example.get("answer", [])
            data_source = example.get("data_source", "unknown")

            # Initialize per-dataset stats
            if data_source not in dataset_stats:
                dataset_stats[data_source] = {"total": 0, "kept": 0}
            dataset_stats[data_source]["total"] += 1

            if isinstance(golden_answers, str):
                golden_answers = [golden_answers]

            # --- 1. Tool call validation (strict) ---
            tool_valid, tool_reason = validate_tool_calls(trajectory)
            if not tool_valid:
                if tool_reason == "no_tool_call":
                    stats["no_tool_call"] += 1
                elif "JSON" in tool_reason:
                    stats["invalid_tool_json"] += 1
                elif "name" in tool_reason:
                    stats["invalid_tool_name"] += 1
                elif "query" in tool_reason.lower():
                    stats["empty_query"] += 1
                else:
                    stats["invalid_tool_json"] += 1
                continue

            num_tool_calls = count_tool_calls(trajectory)

            # --- 2. Must have final answer (complete <answer>...</answer>) ---
            if not has_final_answer(trajectory):
                stats["missing_final_answer"] += 1
                continue

            # --- 3. Answer must be correct (EM) ---
            predicted_answer = extract_final_answer(trajectory)
            if not predicted_answer or not em_check(predicted_answer, golden_answers):
                stats["wrong_answer"] += 1
                continue

            # --- 4. Check for failed tool responses ---
            if has_failed_tool_response(trajectory):
                stats["failed_retrieval"] += 1
                continue

            # --- 5. Validate role sequence ---
            seq_valid, seq_reason = validate_role_sequence(trajectory)
            if not seq_valid:
                if seq_reason == "missing_tool_response":
                    stats["missing_tool_response"] += 1
                elif seq_reason.startswith("orphan") or seq_reason == "tool_without_tool_call":
                    stats["missing_tool_response"] += 1
                else:
                    stats["invalid_role_sequence"] += 1
                continue

            # --- 6. Must not exceed max turns ---
            assistant_turns = sum(1 for t in trajectory if t.get("role") == "assistant")
            if assistant_turns > args.max_turns:
                stats["too_many_turns"] += 1
                continue

            # --- 7. Must not exceed max tool calls ---
            if num_tool_calls > args.max_tool_calls:
                stats["too_many_tool_calls"] += 1
                continue

            # --- 8. Token length check ---
            if not check_token_length(trajectory, tokenizer, SEARCH_TOOL_SCHEMA, args.max_length):
                stats["too_long"] += 1
                continue

            # --- Write output ---
            output_record = {
                "messages": trajectory,
                "tools": json.dumps(SEARCH_TOOL_SCHEMA, ensure_ascii=False),
                "metadata": {
                    "question": question,
                    "answers": golden_answers,
                    "data_source": data_source,
                    "num_tool_calls": num_tool_calls,
                    "em_correct": True,
                    "retrieval_success": True,
                },
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            dataset_stats[data_source]["kept"] += 1

    # -----------------------------------------------------------------------
    # Log summary
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SFT Data Build Summary")
    logger.info("=" * 60)
    logger.info(f"Total input:      {stats['total']}")
    logger.info(f"Kept:             {stats['kept']}  ({stats['kept'] / max(stats['total'], 1) * 100:.1f}%)")
    logger.info(f"Discarded:        {stats['total'] - stats['kept']}")
    logger.info("")

    discard_total = stats["total"] - stats["kept"]
    if discard_total > 0:
        logger.info("Discard reasons:")
        for reason in [
            "no_tool_call",
            "invalid_tool_json",
            "invalid_tool_name",
            "empty_query",
            "failed_retrieval",
            "missing_tool_response",
            "invalid_role_sequence",
            "missing_final_answer",
            "wrong_answer",
            "too_many_turns",
            "too_many_tool_calls",
            "too_long",
        ]:
            count = stats.get(reason, 0)
            if count > 0:
                logger.info(f"  {reason:30s}: {count:5d} ({count / max(discard_total, 1) * 100:.1f}%)")

    logger.info("")
    logger.info("Per-dataset statistics:")
    for ds_name, ds_stat in sorted(dataset_stats.items()):
        keep_pct = ds_stat["kept"] / max(ds_stat["total"], 1) * 100
        logger.info(f"  {ds_name:20s}: {ds_stat['total']:6d} total, {ds_stat['kept']:6d} kept ({keep_pct:.1f}%)")

    # Token length statistics (if tokenizer was used)
    if tokenizer is not None:
        logger.info("")
        logger.info("Note: Token length filtering is active (max_length=%d)", args.max_length)
    else:
        logger.info("")
        logger.info("Note: Token length filtering is DISABLED (no tokenizer provided).")
        logger.info("      Pass --tokenizer_path to enable length checking.")


if __name__ == "__main__":
    main()
