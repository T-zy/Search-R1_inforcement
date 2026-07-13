# Copyright 2025 Search-R1 Reinforcement Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reward function for Search-R1 GRPO training with verl Tool Agent Loop.

Reward design (matching original Search-R1's ``qa_em_format.compute_score_em``):

| Condition                                    | Score         |
|----------------------------------------------|---------------|
| Answer correct + valid format                | 1.0           |
| Answer correct + invalid format              | 0.8           |
| No answer + valid format + retrieval correct | 0.3           |
| No answer + valid format + retrieval wrong   | 0.2           |
| Wrong answer + valid format + retrieval corr.| 0.3           |
| Wrong answer + valid format + retrieval wrong| 0.2           |
| No answer/wrong + invalid format             | 0.0 or 0.1   |

This function is designed to be called from a custom RewardManager
within the verl training pipeline.
"""

import re
import string
import json
from typing import Any, Optional


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers: list[str]) -> bool:
    """Exact match between predicted answer and any golden answer."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return True
    return False


def extract_solution(solution_str: str) -> Optional[str]:
    """Extract the last <answer>...</answer> block from the solution string."""
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if len(matches) <= 1:
        return None
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    """Extract all <information>...</information> blocks."""
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> bool:
    """Check if any retrieved passage contains the golden answer."""
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def is_valid_sequence(text: str) -> tuple[bool, str]:
    """
    Validate the structure/format of the assistant's response.

    Expected sequence:
        <think>...</think>
        <search>...</search>
        <information>...</information>
        <think>...</think>
        ...
        <answer>...</answer>

    Returns:
        (is_valid, reason)
    """
    # Check required tags exist
    has_think = "<think>" in text and "</think>" in text
    has_search = "<search>" in text and "</search>" in text
    has_information = "<information>" in text and "</information>" in text
    has_answer = "<answer>" in text and "</answer>" in text

    if not has_answer:
        return False, "missing_answer_tag"

    # Count tag occurrences (must be balanced)
    tags = {
        "think": (text.count("<think>"), text.count("</think>")),
        "search": (text.count("<search>"), text.count("</search>")),
        "information": (text.count("<information>"), text.count("</information>")),
        "answer": (text.count("<answer>"), text.count("</answer>")),
    }

    for tag_name, (open_count, close_count) in tags.items():
        if open_count != close_count:
            return False, f"unbalanced_{tag_name}_tags"
        if open_count < 1 and tag_name in ("think", "answer"):
            return False, f"missing_{tag_name}_tag"

    # Check sequence order: must start with think, then search, then info
    # Use a simple state machine
    state = "start"
    tokens = re.split(r"(</?think>|</?search>|</?information>|</?answer>)", text)
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token == "<think>":
            if state == "start":
                state = "thinking"
            elif state in ("after_info", "after_search"):
                state = "thinking"
            else:
                return False, f"unexpected_think_in_state_{state}"
        elif token == "</think>":
            if state == "thinking":
                state = "after_think"
            else:
                return False, f"unexpected_endthink_in_state_{state}"
        elif token == "<search>":
            if state in ("after_think", "after_info"):
                state = "searching"
            else:
                return False, f"unexpected_search_in_state_{state}"
        elif token == "</search>":
            if state == "searching":
                state = "after_search"
            else:
                return False, f"unexpected_endsearch_in_state_{state}"
        elif token == "<information>":
            if state == "after_search":
                state = "information"
            else:
                return False, f"unexpected_info_in_state_{state}"
        elif token == "</information>":
            if state == "information":
                state = "after_info"
            else:
                return False, f"unexpected_endinfo_in_state_{state}"
        elif token == "<answer>":
            if state in ("after_think", "after_info", "thinking"):
                state = "answering"
            else:
                return False, f"unexpected_answer_in_state_{state}"
        elif token == "</answer>":
            if state == "answering":
                state = "answered"
            else:
                return False, f"unexpected_endanswer_in_state_{state}"

    if state != "answered":
        return False, f"incomplete_sequence_state_{state}"

    return True, "valid"


def compute_score_em(
    solution_str: str,
    ground_truth: dict[str, Any],
    method: str = "strict",
    structure_format_score: float = 0.2,
    final_format_score: float = 0.1,
    retrieval_score: float = 0.1,
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """
    Compute the reward score for a Search-R1 trajectory.

    Args:
        solution_str: The full assistant response text.
        ground_truth: Dict with 'target' key containing answer(s).
        structure_format_score: Reward for valid structure format (tags).
        final_format_score: Reward for having answer tags at all.
        retrieval_score: Bonus when retrieval contains the answer.
        format_score: Fallback format score.
        score: Maximum reward for correct answer.

    Returns:
        float reward score.
    """
    is_valid_format, format_reason = is_valid_sequence(solution_str)
    retrieval_correct = False

    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth.get("target", []))

    answer = extract_solution(solution_str=solution_str)

    if answer is None:
        # No final answer extracted
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score
            else:
                return structure_format_score
        else:
            return 0.0
    else:
        if em_check(answer, ground_truth.get("target", [])):
            if is_valid_format:
                return score
            else:
                return score - structure_format_score
        elif is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score
            else:
                return structure_format_score
        else:
            return final_format_score
