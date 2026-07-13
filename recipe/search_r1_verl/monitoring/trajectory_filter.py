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
Trajectory anomaly detection and filtering for Search-R1 GRPO training.

This module classifies trajectory anomalies into two categories:
1. **System/Environment anomalies**  -> should be LOSS MASKED (response_mask=0)
2. **Model strategy errors**          -> should be PENALIZED via reward only

For GRPO, masked trajectories are excluded from group mean/std computation
to avoid polluting the baseline. If a group (uid) has fewer than
MIN_VALID_TRAJECTORIES valid trajectories, the entire group is skipped.
"""

import logging
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__file__)


# ---- Anomaly type classification ----

class AnomalyType(str, Enum):
    """Classification of trajectory anomalies."""

    # System/Environment anomalies -> should be loss masked
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_HTTP_ERROR = "tool_http_error"
    TOOL_PARSE_ERROR = "tool_parse_error"
    RETRIEVER_CRASH = "retriever_crash"
    EMPTY_RETRIEVAL_SYSTEM = "empty_retrieval_due_to_system"
    TOOL_RESPONSE_TRUNCATED = "tool_response_truncated"
    SEQUENCE_TRUNCATED = "sequence_truncated"
    ROLLOUT_ENGINE_ERROR = "rollout_engine_error"
    MAX_TOOL_CALLS_EXCEEDED = "max_tool_calls_exceeded"

    # Model strategy errors -> penalize via reward only, do NOT mask
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    INVALID_ANSWER_FORMAT = "invalid_answer_format"
    NO_FINAL_ANSWER = "no_final_answer"
    UNNECESSARY_SEARCH = "unnecessary_search"
    WRONG_ANSWER = "wrong_answer"


# Default configuration: which anomaly types trigger loss masking
DEFAULT_MASK_ON = {
    AnomalyType.TOOL_TIMEOUT,
    AnomalyType.TOOL_HTTP_ERROR,
    AnomalyType.TOOL_PARSE_ERROR,
    AnomalyType.TOOL_RESPONSE_TRUNCATED,
    AnomalyType.SEQUENCE_TRUNCATED,
    AnomalyType.MAX_TOOL_CALLS_EXCEEDED,
}

# Anomaly types that only penalize (no mask)
DEFAULT_PENALIZE_ONLY = {
    AnomalyType.INVALID_TOOL_ARGUMENTS,
    AnomalyType.INVALID_ANSWER_FORMAT,
    AnomalyType.NO_FINAL_ANSWER,
    AnomalyType.WRONG_ANSWER,
}


# ---- Detector functions ----

def detect_tool_anomalies(tool_metrics: dict[str, Any]) -> list[AnomalyType]:
    """Detect anomalies from tool execution metrics."""
    anomalies = []

    if tool_metrics.get("tool/search_timeout", 0) > 0:
        anomalies.append(AnomalyType.TOOL_TIMEOUT)
    if tool_metrics.get("tool/search_exception_type", "none") == "http_error":
        anomalies.append(AnomalyType.TOOL_HTTP_ERROR)
    if tool_metrics.get("tool/search_exception_type", "none") == "parse_error":
        anomalies.append(AnomalyType.TOOL_PARSE_ERROR)
    if tool_metrics.get("tool/search_response_truncated", 0) > 0:
        anomalies.append(AnomalyType.TOOL_RESPONSE_TRUNCATED)

    return anomalies


def detect_rollout_anomalies(
    response_length: int,
    max_response_length: int,
    num_tool_calls: int,
    max_tool_calls: int,
) -> list[AnomalyType]:
    """Detect anomalies from rollout statistics."""
    anomalies = []

    if response_length >= max_response_length:
        anomalies.append(AnomalyType.SEQUENCE_TRUNCATED)
    if num_tool_calls > max_tool_calls:
        anomalies.append(AnomalyType.MAX_TOOL_CALLS_EXCEEDED)

    return anomalies


def detect_format_anomalies(
    has_answer: bool,
    has_valid_format: bool,
    answer_correct: bool,
) -> list[AnomalyType]:
    """Detect format/strategy anomalies from the response text."""
    anomalies = []

    if not has_answer:
        anomalies.append(AnomalyType.NO_FINAL_ANSWER)
    elif not answer_correct:
        anomalies.append(AnomalyType.WRONG_ANSWER)
    if not has_valid_format and has_answer:
        anomalies.append(AnomalyType.INVALID_ANSWER_FORMAT)

    return anomalies


# ---- Filtering logic ----

class TrajectoryFilter:
    """
    Filters trajectories based on detected anomalies.

    For GRPO training:
    - System anomalies -> response_mask = 0 (loss masked)
    - Strategy errors  -> reward stays low (learns from penalty)
    - If a uid group has < MIN_VALID valid trajectories -> skip entire group
    """

    def __init__(
        self,
        mask_on: Optional[set[AnomalyType]] = None,
        min_valid_trajectories: int = 2,
        max_tool_calls: int = 10,
    ):
        self.mask_on = mask_on or DEFAULT_MASK_ON
        self.min_valid_trajectories = min_valid_trajectories
        self.max_tool_calls = max_tool_calls

    def classify_trajectory(
        self,
        tool_metrics: dict[str, Any],
        response_length: int,
        max_response_length: int,
        num_tool_calls: int,
        has_answer: bool,
        has_valid_format: bool,
        answer_correct: bool,
    ) -> tuple[bool, list[AnomalyType]]:
        """
        Classify a single trajectory.

        Returns:
            (should_mask, detected_anomalies)
        """
        anomalies = []

        # Tool-level anomalies
        anomalies.extend(detect_tool_anomalies(tool_metrics))

        # Rollout-level anomalies
        anomalies.extend(detect_rollout_anomalies(
            response_length, max_response_length, num_tool_calls, self.max_tool_calls
        ))

        # Format/strategy anomalies
        anomalies.extend(detect_format_anomalies(has_answer, has_valid_format, answer_correct))

        # Determine whether to mask
        should_mask = any(ano in self.mask_on for ano in anomalies)

        return should_mask, anomalies

    def filter_group(
        self,
        group_trajectories: list[dict[str, Any]],
    ) -> tuple[list[int], list[int], list[dict[str, Any]]]:
        """
        Filter a group of trajectories (same uid, multiple rollouts).

        Args:
            group_trajectories: List of trajectory dicts, each containing:
                - 'tool_metrics': dict
                - 'response_length': int
                - 'max_response_length': int
                - 'num_tool_calls': int
                - 'has_answer': bool
                - 'has_valid_format': bool
                - 'answer_correct': bool
                - 'reward': float

        Returns:
            (valid_indices, masked_indices, skipped_group_info)
            - valid_indices: indices of trajectories to keep in GRPO computation
            - masked_indices: indices of trajectories that were loss masked
            - skipped_group_info: if group is skipped, info dict; else None
        """
        valid_indices = []
        masked_indices = []
        all_anomalies = []

        for idx, traj in enumerate(group_trajectories):
            should_mask, anomalies = self.classify_trajectory(
                tool_metrics=traj.get("tool_metrics", {}),
                response_length=traj.get("response_length", 0),
                max_response_length=traj.get("max_response_length", 1),
                num_tool_calls=traj.get("num_tool_calls", 0),
                has_answer=traj.get("has_answer", False),
                has_valid_format=traj.get("has_valid_format", False),
                answer_correct=traj.get("answer_correct", False),
            )
            all_anomalies.append(anomalies)

            if should_mask:
                masked_indices.append(idx)
            else:
                valid_indices.append(idx)

        # Check minimum valid trajectories requirement
        skipped_info = None
        if len(valid_indices) < self.min_valid_trajectories:
            skipped_info = {
                "n_total": len(group_trajectories),
                "n_valid": len(valid_indices),
                "n_masked": len(masked_indices),
                "min_required": self.min_valid_trajectories,
                "anomalies_per_traj": [str(a) for a in all_anomalies],
            }
            # If group is skipped, all trajectories are invalid
            valid_indices = []
            masked_indices = list(range(len(group_trajectories)))

        return valid_indices, masked_indices, skipped_info
