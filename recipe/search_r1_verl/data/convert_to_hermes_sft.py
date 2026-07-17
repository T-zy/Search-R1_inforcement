#!/usr/bin/env python3
"""
Convert old Search-R1 trajectory format (<search> tags) to Hermes function call format.

Old format:
    assistant: Let me search. <search>query text</search>
    tool: <information>...</information>
    assistant: <answer>...</answer>

Hermes format:
    assistant: Let me search.
               <tool_call>{"name": "search", "arguments": {"query": "query text"}}</tool_call>
    tool: <information>...</information>
    assistant: <answer>...</answer>

Also supports already-converted trajectories (idempotent conversion).

Usage:
    python convert_to_hermes_sft.py \\
        --input /path/to/raw_trajectories.jsonl \\
        --output /path/to/hermes_trajectories.jsonl

Input JSONL format (from generate_teacher_trajectories.py):
    {
        "question": "...",
        "system_prompt": "...",
        "trajectory": [
            {"role": "user", "content": "Question ..."},
            {"role": "assistant", "content": "...<search>query</search>..."},
            {"role": "tool", "content": "<information>...</information>"},
            {"role": "assistant", "content": "...<answer>...</answer>..."}
        ],
        "answer": ["golden_answer"],
        "data_source": "nq",
        "retrieval_success": true,
        "trajectory_valid": true,
        "num_tool_calls": 1
    }

Output JSONL format:
    Same structure, but all <search> tags in assistant messages
    are converted to <tool_call> Hermes format.
"""

import argparse
import json
import logging
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Pattern for old-style <search> tags
SEARCH_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)


def convert_search_to_tool_call(text: str) -> str:
    """Convert <search>query</search> to <tool_call> Hermes format.

    Handles three cases:
    1. query is already a JSON object (e.g., {"query": "..."}) — wrap directly
    2. query is raw text — wrap as {"query": "raw text"}
    3. query is a JSON string with name/arguments — use as-is
    """
    def _replace(match):
        query_content = match.group(1).strip()

        # Case 1: query is already a full tool_call JSON
        try:
            parsed = json.loads(query_content)
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                # Already in {name, arguments} format
                tool_call = json.dumps(parsed, ensure_ascii=False)
                return f"<tool_call>\n{tool_call}\n</tool_call>"
            elif isinstance(parsed, dict):
                # JSON object — wrap as arguments
                tool_call = json.dumps(
                    {"name": "search", "arguments": parsed},
                    ensure_ascii=False,
                )
                return f"<tool_call>\n{tool_call}\n</tool_call>"
            else:
                # JSON but not a dict (e.g., string) — wrap as query
                tool_call = json.dumps(
                    {"name": "search", "arguments": {"query": str(parsed)}},
                    ensure_ascii=False,
                )
                return f"<tool_call>\n{tool_call}\n</tool_call>"
        except json.JSONDecodeError:
            pass

        # Case 2: raw text query
        tool_call = json.dumps(
            {"name": "search", "arguments": {"query": query_content}},
            ensure_ascii=False,
        )
        return f"<tool_call>\n{tool_call}\n</tool_call>"

    return SEARCH_PATTERN.sub(_replace, text)


def is_already_hermes(text: str) -> bool:
    """Check if text already contains Hermes <tool_call> format."""
    return "<tool_call>" in text


def convert_trajectory(trajectory: list[dict]) -> list[dict]:
    """Convert all assistant messages in a trajectory to Hermes format."""
    new_trajectory = []
    for turn in trajectory:
        new_turn = dict(turn)
        if turn.get("role") == "assistant":
            content = turn.get("content", "")
            if not is_already_hermes(content):
                new_turn["content"] = convert_search_to_tool_call(content)
            # If already Hermes, leave as-is (idempotent)
        new_trajectory.append(new_turn)
    return new_trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Convert old <search> format to Hermes <tool_call> format"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with trajectories (old or mixed format)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file with Hermes format trajectories")
    parser.add_argument("--stats", action="store_true", default=True,
                        help="Print conversion statistics")
    args = parser.parse_args()

    total = 0
    converted = 0
    already_hermes = 0
    skipped = 0
    errors = 0

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                skipped += 1
                continue

            total += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON at line ~{total}: {e}")
                errors += 1
                continue

            trajectory = data.get("trajectory", [])
            if not trajectory:
                skipped += 1
                continue

            # Check if any assistant message needs conversion
            needs_conversion = any(
                turn.get("role") == "assistant"
                and not is_already_hermes(turn.get("content", ""))
                and "<search>" in turn.get("content", "")
                for turn in trajectory
            )

            if needs_conversion:
                data["trajectory"] = convert_trajectory(trajectory)
                converted += 1
            else:
                already_hermes += 1

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    if args.stats:
        logger.info("=" * 50)
        logger.info("Conversion Statistics")
        logger.info("=" * 50)
        logger.info(f"Total records:        {total}")
        logger.info(f"Converted:            {converted}  ({converted / max(total, 1) * 100:.1f}%)")
        logger.info(f"Already Hermes:       {already_hermes}  ({already_hermes / max(total, 1) * 100:.1f}%)")
        logger.info(f"Skipped (empty):      {skipped}")
        logger.info(f"Errors:               {errors}")
        logger.info(f"Output:               {args.output}")

    print(f"Converted {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
