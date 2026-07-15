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
Reward function for Search-R1 GRPO training with verl Tool Agent Loop (v2).

Reward design (最终执行方案 v1.0):

| Condition                                         | Score  |
|---------------------------------------------------|--------|
| HotpotQA search + correct answer                  | 1.00   |
| NQ search + correct answer                        | 0.95   |
| NQ no search + correct answer                     | 0.90   |
| HotpotQA no search + correct answer               | 0.70   |
| Search + retrieval contains answer + wrong answer  | 0.30   |
| Search + wrong answer                              | 0.15   |
| No search + wrong answer + has <answer> tag       | -0.10  |
| No <answer> tag                                   | 0.00   |

Key design principles:
  1. Searching and being correct > not searching and being correct
  2. Multi-hop datasets (hotpotqa) penalise zero-search
  3. Reward returns dict for per-component wandb logging
  4. No dependency on <search> tags (Hermes tool-call format)
  5. Search detection via <information> blocks
"""

import os
import re
import string
from typing import Any

# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    """Lower text, remove articles, punctuation and extra whitespace."""

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
    """Exact match with normalisation."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(ans) for ans in golden_answers)


def extract_solution(solution_str: str):
    """
    Extract the last <answer>...</answer> block.

    Returns None if no <answer> tag is found (fixed: was incorrectly
    returning None when exactly one match existed).
    """
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    """Extract all <information>...</information> blocks."""
    return [
        block.strip()
        for block in re.findall(r"<information>(.*?)</information>", text, re.DOTALL)
    ]


def extract_search_info(solution_str: str) -> tuple[int, bool]:
    """
    Detect search activity from <information> blocks (XML fallback).

    In verl Hermes Tool Agent format, the model does NOT emit <search>
    tags (it uses function-call instead), so we detect search by counting
    <information> blocks injected by the tool response.

    Returns:
        (num_search_calls, has_successful_search)
    """
    blocks = extract_information_blocks(solution_str)
    if not blocks:
        return 0, False
    success = any("Search failed:" not in b for b in blocks)
    return len(blocks), success


def retrieval_contains_answer(information_blocks: list[str], targets: list[str]) -> bool:
    """Check if any retrieved passage contains any target answer."""
    for block in information_blocks:
        norm_block = normalize_answer(block)
        for target in targets:
            if normalize_answer(target) in norm_block:
                return True
    return False


# ---------------------------------------------------------------------------
# Tool metrics extraction (primary) with XML fallback
# ---------------------------------------------------------------------------

def extract_tool_state(
    solution_str: str,
    extra_info: dict | None,
) -> dict:
    """
    Extract search tool state from structured metrics (primary) or XML fallback.

    Structured Tool metrics take precedence when available.
    XML <information> block parsing is used as fallback.
    """
    extra_info = extra_info or {}

    information_blocks = extract_information_blocks(solution_str)
    xml_search_count = len(information_blocks)
    xml_success = any(
        block.strip() and "Search failed:" not in block
        for block in information_blocks
    )

    metrics_available = "tool/search_called" in extra_info

    if metrics_available:
        num_search_calls = int(extra_info.get("tool/search_called", 0))
        search_success_count = int(extra_info.get("tool/search_success", 0))
        search_failed_count = int(extra_info.get("tool/search_failed", 0))
        search_timeout_count = int(extra_info.get("tool/search_timeout", 0))
        search_num_docs = int(extra_info.get("tool/search_num_docs", 0))
    else:
        num_search_calls = xml_search_count
        search_success_count = int(xml_success)
        search_failed_count = int(xml_search_count > 0 and not xml_success)
        search_timeout_count = 0
        search_num_docs = int(xml_success)

    has_successful_search = (
        num_search_calls > 0
        and search_success_count > 0
        and search_num_docs > 0
    )

    all_searches_successful = (
        num_search_calls > 0
        and search_success_count == num_search_calls
        and search_failed_count == 0
    )

    return {
        "metrics_available": metrics_available,
        "information_blocks": information_blocks,
        "num_search_calls": num_search_calls,
        "search_success_count": search_success_count,
        "search_failed_count": search_failed_count,
        "search_timeout_count": search_timeout_count,
        "search_num_docs": search_num_docs,
        "has_successful_search": has_successful_search,
        "all_searches_successful": all_searches_successful,
    }


# ---- Pipeline debug logging ----
_PIPELINE_DEBUG_PRINTED = False


# ---------------------------------------------------------------------------
# Main reward function (verl-native interface)
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> dict:
    """
    Compute reward for a single GRPO trajectory.

    Tool metrics from structured extra_info are used as primary source.
    XML <information> block parsing is used as fallback.

    Args:
        data_source: Dataset identifier (e.g. "nq", "hotpotqa").
        solution_str: Full assistant response text (including tool responses).
        ground_truth: Must contain ``{"target": ["answer1", ...]}``.
        extra_info: Optional metadata from ToolAgentLoop (may include
            ``tool/search_called``, ``tool/search_success``, etc.).

    Returns:
        dict with ``"score"`` (float) as the primary reward, plus per-component
        metrics that verl automatically logs to wandb.
    """
    targets = ground_truth.get("target", [])
    if isinstance(targets, str):
        targets = [targets]

    # ---- Answer extraction ----
    answer = extract_solution(solution_str)
    has_answer = answer is not None
    answer_correct = has_answer and em_check(answer, targets)

    # ---- Search detection (metrics primary, XML fallback) ----
    tool_state = extract_tool_state(solution_str, extra_info)
    has_successful_search = tool_state["has_successful_search"]
    num_search_calls = tool_state["num_search_calls"]

    # ---- Pipeline debug logging (one-shot) ----
    global _PIPELINE_DEBUG_PRINTED
    if (
        os.getenv("SEARCH_R1_DEBUG_PIPELINE", "0") == "1"
        and not _PIPELINE_DEBUG_PRINTED
    ):
        print(f"[Search-R1 pipeline] data_source={data_source}")
        print(f"[Search-R1 pipeline] extra_info keys={sorted(extra_info.keys()) if extra_info else 'EMPTY'}")
        print(f"[Search-R1 pipeline] tool_state={tool_state}")
        _PIPELINE_DEBUG_PRINTED = True

    # ---- Retrieval evidence check ----
    info_blocks = tool_state["information_blocks"]
    retrieval_ok = retrieval_contains_answer(info_blocks, targets) if info_blocks else False

    # ---- Reward computation ----
    norm_source = str(data_source).lower()
    is_multi_hop = norm_source in ("hotpotqa", "2wikimultihopqa", "musique", "bamboogle")

    reward = 0.0

    if answer_correct:
        reward += 0.8

    if has_answer:
        reward += 0.1

    if has_successful_search:
        reward += 0.05

    if retrieval_ok:
        reward += 0.05

    # 注意：失败搜索不能绕过多跳搜索惩罚
    if is_multi_hop and not has_successful_search:
        reward -= 0.2

    reward = max(-0.2, min(1.0, reward))

    missing_penalty = float(is_multi_hop and not has_successful_search)

    # ---- Return as dict for wandb logging ----
    return {
        "score": float(reward),
        "answer_em": float(answer_correct),
        "has_final_answer": float(has_answer),
        "tool_metrics_available": float(tool_state["metrics_available"]),
        "num_search_calls": float(tool_state["num_search_calls"]),
        "search_success_count": float(tool_state["search_success_count"]),
        "search_failed_count": float(tool_state["search_failed_count"]),
        "search_timeout_count": float(tool_state["search_timeout_count"]),
        "search_num_docs": float(tool_state["search_num_docs"]),
        "has_successful_search": float(tool_state["has_successful_search"]),
        "all_searches_successful": float(tool_state["all_searches_successful"]),
        "retrieval_correct": float(retrieval_ok),
        "missing_successful_search_penalty": missing_penalty,
    }
