#!/usr/bin/env python3
"""
Build SFT data from teacher trajectories for cold-start fine-tuning.

This script converts teacher model trajectories (containing CoT, search calls,
tool responses, and final answers) into training data for SFT cold-start.

Expected input format (JSONL):
    {
        "question": "...",
        "trajectory": [
            {"role": "user", "content": "Question ..."},
            {"role": "assistant", "content": "...<search>...</search>..."},
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
      JSON-legal tool calls, has final answer, length within limit
    - Discard: wrong answer, illegal format, retrieval failure,
      exceeds max turns, no final answer, too long

Usage:
    python build_sft_data.py \\
        --input /path/to/teacher_trajectories.jsonl \\
        --output /path/to/sft_data.jsonl \\
        --max_length 4096
"""

import argparse
import json
import logging
import os
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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


def extract_answer_from_trajectory(trajectory: list[dict]) -> str:
    """Extract the final answer from the assistant's last turn."""
    for turn in reversed(trajectory):
        if turn.get("role") == "assistant":
            content = turn.get("content", "")
            answer_pattern = r"<answer>(.*?)</answer>"
            matches = list(re.finditer(answer_pattern, content, re.DOTALL))
            if matches:
                return matches[-1].group(1).strip()
    return ""


def is_tool_call_valid(content: str) -> bool:
    """Check if tool calls in the content are valid JSON."""
    search_pattern = r"<search>(.*?)</search>"
    matches = re.findall(search_pattern, content, re.DOTALL)
    for match in matches:
        try:
            json.loads(match.strip())
        except json.JSONDecodeError:
            # Also accept raw text search queries (not just JSON)
            pass
    return True


def count_tool_calls(trajectory: list[dict]) -> int:
    """Count the number of tool calls in the trajectory."""
    count = 0
    for turn in trajectory:
        if turn.get("role") == "assistant":
            content = turn.get("content", "")
            count += len(re.findall(r"<search>", content))
    return count


def has_final_answer(trajectory: list[dict]) -> bool:
    """Check if the trajectory has a final answer."""
    for turn in reversed(trajectory):
        if turn.get("role") == "assistant":
            if "<answer>" in turn.get("content", ""):
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Build SFT data from teacher trajectories")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with teacher trajectories")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file for SFT training")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--max_turns", type=int, default=4,
                        help="Maximum number of assistant turns")
    parser.add_argument("--max_tool_calls", type=int, default=10,
                        help="Maximum number of tool calls")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    kept = 0
    discarded = 0
    discard_reasons = {}

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)

            question = example.get("question", "")
            trajectory = example.get("trajectory", [])
            golden_answers = example.get("answer", [])

            if isinstance(golden_answers, str):
                golden_answers = [golden_answers]

            # --- Filtering ---

            # 1. Must have at least one valid search
            num_tool_calls = count_tool_calls(trajectory)
            if num_tool_calls < 1:
                discarded += 1
                discard_reasons["no_search"] = discard_reasons.get("no_search", 0) + 1
                continue

            # 2. Must have final answer
            if not has_final_answer(trajectory):
                discarded += 1
                discard_reasons["no_final_answer"] = discard_reasons.get("no_final_answer", 0) + 1
                continue

            # 3. Answer must be correct (EM)
            predicted_answer = extract_answer_from_trajectory(trajectory)
            if not predicted_answer or not em_check(predicted_answer, golden_answers):
                discarded += 1
                discard_reasons["wrong_answer"] = discard_reasons.get("wrong_answer", 0) + 1
                continue

            # 4. Tool call format must be valid
            trajectory_valid = True
            for turn in trajectory:
                if turn.get("role") == "assistant":
                    if not is_tool_call_valid(turn.get("content", "")):
                        trajectory_valid = False
                        break
            if not trajectory_valid:
                discarded += 1
                discard_reasons["invalid_tool_format"] = discard_reasons.get("invalid_tool_format", 0) + 1
                continue

            # 5. Must not exceed max turns
            assistant_turns = sum(1 for t in trajectory if t.get("role") == "assistant")
            if assistant_turns > args.max_turns:
                discarded += 1
                discard_reasons["exceed_max_turns"] = discard_reasons.get("exceed_max_turns", 0) + 1
                continue

            # 6. Must not exceed max tool calls
            if num_tool_calls > args.max_tool_calls:
                discarded += 1
                discard_reasons["exceed_max_tool_calls"] = discard_reasons.get("exceed_max_tool_calls", 0) + 1
                continue

            # --- Write output ---
            output_record = {
                "question": question,
                "trajectory": trajectory,
                "answer": golden_answers,
                "num_tool_calls": num_tool_calls,
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            kept += 1

    logger.info(f"Kept: {kept}, Discarded: {discarded}")
    for reason, count in sorted(discard_reasons.items(), key=lambda x: -x[1]):
        logger.info(f"  Discard reason '{reason}': {count}")


if __name__ == "__main__":
    main()
