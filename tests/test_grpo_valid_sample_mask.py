"""
Unit tests for GRPO advantage computation with valid_sample_mask.

Tests the non-vectorized compute_grpo_outcome_advantage() function.

Group A: 4 trajectories, 3 valid + 1 invalid
Group B: 4 trajectories, 1 valid + 3 invalid
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def _make_inputs(
    rewards: list[list[float]],
    masks: list[list[float]],
    uids: list[str],
    valid_mask: list[bool] | None = None,
):
    """Helper to create test tensors."""
    device = torch.device("cpu")
    token_level_rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    response_mask = torch.tensor(masks, dtype=torch.float32, device=device)
    index = np.array(uids, dtype=object)
    if valid_mask is not None:
        valid_sample_mask = np.array(valid_mask, dtype=bool)
    else:
        valid_sample_mask = None
    return token_level_rewards, response_mask, index, valid_sample_mask


def test_group_valid_trajectories_used():
    """
    Group A: 4 trajectories, 3 valid.
    The 3 valid scores should be used for mean/std.
    The invalid trajectory should have advantage = 0.
    """
    # Scores: valid=[5, 4, 3], invalid=[0]
    rewards = [[5.0], [4.0], [3.0], [0.0]]
    masks = [[1.0], [1.0], [1.0], [1.0]]
    uids = ["A", "A", "A", "A"]
    valid = [True, True, True, False]

    tlr, rmask, idx, vmask = _make_inputs(rewards, masks, uids, valid)
    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=tlr,
        response_mask=rmask,
        index=idx,
        valid_sample_mask=vmask,
    )

    # Valid trajectories should have non-zero advantages
    assert advantages[0, 0].item() != 0.0
    assert advantages[1, 0].item() != 0.0
    assert advantages[2, 0].item() != 0.0

    # Invalid trajectory should have advantage = 0
    assert advantages[3, 0].item() == 0.0, f"Expected 0, got {advantages[3, 0].item()}"

    # Verify the valid advantage pattern: highest reward gets positive advantage, lowest gets negative
    # mean = (5+4+3)/3 = 4, std ≈ 0.816
    # (5-4)/0.816 ≈ 1.225, (4-4)/0.816 = 0, (3-4)/0.816 ≈ -1.225
    adv_0 = advantages[0, 0].item()  # score 5 -> positive
    adv_2 = advantages[2, 0].item()  # score 3 -> negative
    assert adv_0 > 0, f"Expected positive advantage for score 5, got {adv_0}"
    assert adv_2 < 0, f"Expected negative advantage for score 3, got {adv_2}"
    # The sum of valid advantages should be close to 0 (within group normalization)
    print(f"✅ Group A valid advantages: {advantages[0,0].item():.4f}, {advantages[1,0].item():.4f}, {advantages[2,0].item():.4f}, invalid={advantages[3,0].item():.4f}")


def test_group_dropped_when_less_than_2_valid():
    """
    Group B: 4 trajectories, only 1 valid.
    Entire group should have advantage = 0.
    """
    rewards = [[10.0], [0.0], [0.0], [0.0]]
    masks = [[1.0], [1.0], [1.0], [1.0]]
    uids = ["B", "B", "B", "B"]
    valid = [True, False, False, False]

    tlr, rmask, idx, vmask = _make_inputs(rewards, masks, uids, valid)
    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=tlr,
        response_mask=rmask,
        index=idx,
        valid_sample_mask=vmask,
    )

    # All advantages must be 0 for the entire dropped group
    for i in range(4):
        assert advantages[i, 0].item() == 0.0, f"Expected 0 for trajectory {i}, got {advantages[i, 0].item()}"
    print(f"✅ Group B (dropped) all advantages zero: {advantages[:, 0].tolist()}")


def test_mixed_groups():
    """
    Group A: 4 trajs, 3 valid -> normal GRPO
    Group B: 4 trajs, 1 valid -> dropped (all zero)
    Group C: 4 trajs, 4 valid -> normal GRPO
    """
    rewards = [
        [5.0], [4.0], [3.0], [0.0],   # Group A
        [10.0], [0.0], [0.0], [0.0],   # Group B (dropped)
        [8.0], [6.0], [4.0], [2.0],    # Group C
    ]
    masks = [[1.0]] * 12
    uids = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    valid = [True, True, True, False] + [True, False, False, False] + [True, True, True, True]

    tlr, rmask, idx, vmask = _make_inputs(rewards, masks, uids, valid)
    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=tlr,
        response_mask=rmask,
        index=idx,
        valid_sample_mask=vmask,
    )

    # Group A: 3 valid, 1 invalid. Invalid should be 0, valid should have varied advantages
    assert advantages[3, 0].item() == 0.0, f"Group A invalid should be 0, got {advantages[3, 0].item()}"
    assert advantages[0, 0].item() > 0, f"Group A highest reward should be positive, got {advantages[0, 0].item()}"
    assert advantages[2, 0].item() < 0, f"Group A lowest valid reward should be negative, got {advantages[2, 0].item()}"

    # Group B: all zero (dropped)
    for i in range(4, 8):
        assert advantages[i, 0].item() == 0.0, f"Group B traj {i} should be 0, got {advantages[i, 0].item()}"

    # Group C: all 4 valid, should have varied advantages
    adv_c = [advantages[i, 0].item() for i in range(8, 12)]
    assert abs(sum(adv_c)) < 1e-5, f"Group C advantages should sum to ~0, got {sum(adv_c)}"
    assert adv_c[0] > adv_c[1] > adv_c[2] > adv_c[3], f"Group C advantages should be sorted, got {adv_c}"

    print(f"✅ Mixed groups test passed")
    print(f"  Group A valid: {[advantages[i,0].item() for i in range(4)]}")
    print(f"  Group B (dropped): {[advantages[i,0].item() for i in range(4,8)]}")
    print(f"  Group C: {adv_c}")


def test_no_valid_mask_all_valid():
    """When no valid_sample_mask is provided, all trajectories are valid (original behaviour)."""
    rewards = [[5.0], [3.0], [1.0], [0.0]]
    masks = [[1.0], [1.0], [1.0], [1.0]]
    uids = ["D", "D", "D", "D"]

    tlr, rmask, idx, _ = _make_inputs(rewards, masks, uids, None)
    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=tlr,
        response_mask=rmask,
        index=idx,
        valid_sample_mask=None,
    )

    # All should participate in GRPO normalization
    advs = [advantages[i, 0].item() for i in range(4)]
    assert abs(sum(advs)) < 1e-5, f"Advantages should sum to ~0, got {sum(advs)}"
    assert advs[0] > advs[1] > advs[2] > advs[3]
    print(f"✅ No valid mask test passed: {advs}")


if __name__ == "__main__":
    test_group_valid_trajectories_used()
    test_group_dropped_when_less_than_2_valid()
    test_mixed_groups()
    test_no_valid_mask_all_valid()
    print("✅✅✅ All GRPO mask tests passed!")
