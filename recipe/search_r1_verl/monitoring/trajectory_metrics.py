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
Trajectory metrics aggregation for wandb logging.

Provides functions to aggregate per-step trajectory metrics into
scalars suitable for console and wandb logging.
"""

import logging
from collections import Counter

logger = logging.getLogger(__name__)
from typing import Any, Optional

from recipe.search_r1_verl.monitoring.trajectory_filter import AnomalyType


def aggregate_trajectory_metrics(
    all_anomalies: list[list[AnomalyType]],
    all_valid_flags: list[bool],
    all_masked_flags: list[bool],
    tool_latencies: Optional[list[float]] = None,
    tool_success_flags: Optional[list[bool]] = None,
    num_turns_list: Optional[list[int]] = None,
    search_calls_list: Optional[list[int]] = None,
    rewards: Optional[list[float]] = None,
) -> dict[str, float]:
    """
    Aggregate per-trajectory metrics into summary scalars.

    Args:
        all_anomalies: Anomalies detected per trajectory.
        all_valid_flags: Whether each trajectory is valid (not masked).
        all_masked_flags: Whether each trajectory was loss masked.
        tool_latencies: Latency of each tool call in ms.
        tool_success_flags: Whether each tool call succeeded.
        num_turns_list: Number of turns per trajectory.
        search_calls_list: Number of search calls per trajectory.
        rewards: Reward per trajectory.

    Returns:
        Dict of aggregated metrics suitable for wandb logging.
    """
    n_total = len(all_anomalies)
    if n_total == 0:
        return {}

    n_valid = sum(all_valid_flags)
    n_masked = sum(all_masked_flags)
    n_invalid = n_total - n_valid

    # Count anomaly types
    anomaly_counter: Counter = Counter()
    for traj_anomalies in all_anomalies:
        for ano in traj_anomalies:
            anomaly_counter[ano.value] += 1

    metrics = {
        # Core rates
        "trajectory/valid_rate": n_valid / max(n_total, 1),
        "trajectory/masked_rate": n_masked / max(n_total, 1),
        "trajectory/invalid_rate": n_invalid / max(n_total, 1),
        "trajectory/total": n_total,
        "trajectory/valid": n_valid,
        "trajectory/masked": n_masked,
    }

    # Anomaly rates
    for anomaly_type in AnomalyType:
        key = f"trajectory/{anomaly_type.value}_rate"
        metrics[key] = anomaly_counter.get(anomaly_type.value, 0) / max(n_total, 1)

    # Tool metrics: latency
    n_lat = 0
    if tool_latencies is not None and len(tool_latencies) > 0:
        sorted_latencies = sorted(float(v) for v in tool_latencies)
        n_lat = len(sorted_latencies)
        metrics["tool/search_latency_ms_mean"] = sum(sorted_latencies) / n_lat
        metrics["tool/search_latency_ms_p50"] = sorted_latencies[n_lat // 2]
        p95_index = min(n_lat - 1, int(n_lat * 0.95))
        metrics["tool/search_latency_ms_p95"] = sorted_latencies[p95_index]
        metrics["tool/search_latency_ms_max"] = sorted_latencies[-1]
        metrics["tool/search_calls_total"] = n_lat

    # Tool metrics: success rate
    if tool_success_flags is not None and len(tool_success_flags) > 0:
        n_success = sum(bool(v) for v in tool_success_flags)
        metrics["tool/search_success_rate"] = n_success / len(tool_success_flags)
        metrics["tool/search_success"] = n_success
        metrics["tool/search_failed"] = len(tool_success_flags) - n_success

    # Rollout metrics
    if num_turns_list is not None and len(num_turns_list) > 0:
        metrics["rollout/turns_mean"] = sum(num_turns_list) / max(len(num_turns_list), 1)
        metrics["rollout/turns_max"] = max(num_turns_list)

    if search_calls_list is not None and len(search_calls_list) > 0:
        metrics["rollout/search_calls_mean"] = sum(search_calls_list) / max(len(search_calls_list), 1)

    # Reward metrics
    if rewards is not None and len(rewards) > 0:
        valid_rewards = [r for r, v in zip(rewards, all_valid_flags) if v]
        if valid_rewards:
            metrics["reward/mean"] = sum(valid_rewards) / len(valid_rewards)
            metrics["reward/min"] = min(valid_rewards)
            metrics["reward/max"] = max(valid_rewards)
            metrics["reward/valid_count"] = len(valid_rewards)

    return metrics


def log_trajectory_summary(metrics: dict[str, float], step: int) -> None:
    """Log a human-readable summary of trajectory metrics."""
    lines = [
        f"=== Trajectory Metrics [Step {step}] ===",
        f"  Valid rate:       {metrics.get('trajectory/valid_rate', 0):.3f}",
        f"  Masked rate:      {metrics.get('trajectory/masked_rate', 0):.3f}",
        f"  Tool success rate: {metrics.get('tool/search_success_rate', 0):.3f}",
        f"  Tool latency p50:  {metrics.get('tool/search_latency_ms_p50', 0):.1f}ms",
        f"  Tool latency p95:  {metrics.get('tool/search_latency_ms_p95', 0):.1f}ms",
        f"  Avg turns:         {metrics.get('rollout/turns_mean', 0):.1f}",
        f"  Avg search calls:  {metrics.get('rollout/search_calls_mean', 0):.1f}",
        f"  Avg reward:        {metrics.get('reward/mean', 0):.3f}",
    ]

    # Add anomaly rates
    anomaly_keys = [k for k in metrics if k.startswith("trajectory/") and k.endswith("_rate")
                    and k not in ("trajectory/valid_rate", "trajectory/masked_rate", "trajectory/invalid_rate")]
    for key in sorted(anomaly_keys):
        if metrics[key] > 0:
            # Extract anomaly name from key
            anomaly_name = key[len("trajectory/"):-len("_rate")]
            lines.append(f"  Anomaly {anomaly_name}: {metrics[key]:.4f}")

    logger.info("\n".join(lines))



