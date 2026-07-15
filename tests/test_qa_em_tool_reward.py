"""
Unit tests for the Search-R1 reward function (qa_em_tool_reward.py).

Covers:
- HotpotQA reward matrix (12 cases)
- NQ reward
- extract_solution() correctness
- Tool metrics override XML fallback
- Empty search results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from recipe.search_r1_verl.rewards.qa_em_tool_reward import (
    compute_score,
    extract_solution,
    extract_information_blocks,
    extract_tool_state,
    normalize_answer,
    em_check,
)


def _make_ground_truth(targets):
    return {"target": targets}


def test_hotpot_search_correct():
    """HotpotQA: successful search + evidence contains answer + correct answer = 1.0"""
    solution = (
        "<information>\nDoc 1(Title: Paris): Paris is the capital of France.\n</information>\n"
        "<answer>Paris</answer>"
    )
    gt = _make_ground_truth(["Paris"])
    result = compute_score("hotpotqa", solution, gt)
    assert result["score"] == pytest.approx(1.0), f"Expected 1.0, got {result['score']}"
    assert result["answer_em"] == 1.0
    assert result["has_successful_search"] == 1.0
    assert result["retrieval_correct"] == 1.0


def test_hotpot_search_correct_no_evidence():
    """HotpotQA: successful search + correct answer but evidence doesn't contain answer = 0.95"""
    solution = (
        "<information>\nDoc 1(Title: London): London is the capital of UK.\n</information>\n"
        "<answer>Paris</answer>"
    )
    gt = _make_ground_truth(["Paris"])
    result = compute_score("hotpotqa", solution, gt)
    assert result["score"] == pytest.approx(0.95), f"Expected 0.95, got {result['score']}"
    assert result["answer_em"] == 1.0
    assert result["has_successful_search"] == 1.0
    assert result["retrieval_correct"] == 0.0


def test_hotpot_no_search_correct():
    """HotpotQA: no search but correct answer = 0.70"""
    solution = "<answer>Paris</answer>"
    gt = _make_ground_truth(["Paris"])
    result = compute_score("hotpotqa", solution, gt)
    assert result["score"] == pytest.approx(0.70), f"Expected 0.70, got {result['score']}"
    assert result["answer_em"] == 1.0
    assert result["has_successful_search"] == 0.0
    assert result["missing_successful_search_penalty"] == 1.0


def test_hotpot_search_failed_correct():
    """HotpotQA: search failed but correct answer = 0.70 (failed search doesn't reduce correct answer reward)"""
    solution = (
        "<information>\nSearch failed: No relevant documents were returned.\n</information>\n"
        "<answer>Paris</answer>"
    )
    gt = _make_ground_truth(["Paris"])
    result = compute_score("hotpotqa", solution, gt)
    assert result["score"] == pytest.approx(0.70), f"Expected 0.70, got {result['score']}"
    assert result["answer_em"] == 1.0
    assert result["has_successful_search"] == 0.0
    assert result["missing_successful_search_penalty"] == 1.0


def test_hotpot_search_evidence_wrong_answer():
    """HotpotQA: successful search + evidence contains answer + wrong answer but has answer tag = 0.20"""
    solution = (
        "<information>\nDoc 1(Title: Paris): Paris is the capital of France.\n</information>\n"
        "<answer>London</answer>"
    )
    gt = _make_ground_truth(["Paris"])
    result = compute_score("hotpotqa", solution, gt)
    assert result["score"] == pytest.approx(0.20), f"Expected 0.20, got {result['score']}"
    assert result["answer_em"] == 0.0
    assert result["has_final_answer"] == 1.0
    assert result["has_successful_search"] == 1.0
    assert result["retrieval_correct"] == 1.0


def test_hotpot_search_wrong_answer():
    """HotpotQA: successful search + wrong answer but has answer tag = 0.15"""
    solution = (
        "<information>\nDoc 1(Title: London): London is the capital of UK.\n</information>\n"
        "<answer>Berlin</answer>"
    )
    gt = _make_ground_truth(["Paris"])
    result = compute_score("hotpotqa", solution, gt)
    assert result["score"] == pytest.approx(0.15), f"Expected 0.15, got {result['score']}"
    assert result["answer_em"] == 0.0
    assert result["has_final_answer"] == 1.0
    assert result["has_successful_search"] == 1.0
    assert result["retrieval_correct"] == 0.0


def test_hotpot_no_search_wrong_answer():
    """HotpotQA: no search, wrong answer, has answer tag = -0.10"""
    solution = "<answer>Berlin</answer>"
    gt = _make_ground_truth(["Paris"])
    result = compute_score("hotpotqa", solution, gt)
    assert result["score"] == pytest.approx(-0.10), f"Expected -0.10, got {result['score']}"
    assert result["has_final_answer"] == 1.0
    assert result["has_successful_search"] == 0.0


def test_hotpot_no_answer():
    """HotpotQA: no answer tag, no successful search = -0.20"""
    solution = "I don't know the answer."
    gt = _make_ground_truth(["Paris"])
    result = compute_score("hotpotqa", solution, gt)
    assert result["score"] == pytest.approx(-0.20), f"Expected -0.20, got {result['score']}"
    assert result["has_final_answer"] == 0.0
    assert result["has_successful_search"] == 0.0


def test_nq_no_search_correct():
    """NQ: no search but correct answer = 0.90"""
    solution = "<answer>Paris</answer>"
    gt = _make_ground_truth(["Paris"])
    result = compute_score("nq", solution, gt)
    assert result["score"] == pytest.approx(0.90), f"Expected 0.90, got {result['score']}"
    assert result["answer_em"] == 1.0
    assert result["has_successful_search"] == 0.0
    assert result["missing_successful_search_penalty"] == 0.0  # NQ is not multi-hop


def test_nq_search_correct():
    """NQ: successful search + correct answer + evidence contains answer = 1.0"""
    solution = (
        "<information>\nDoc 1(Title: Paris): Paris is the capital of France.\n</information>\n"
        "<answer>Paris</answer>"
    )
    gt = _make_ground_truth(["Paris"])
    result = compute_score("nq", solution, gt)
    # 0.8 (correct) + 0.1 (has answer) + 0.05 (search success) + 0.05 (retrieval correct) = 1.0
    assert result["score"] == pytest.approx(1.0), f"Expected 1.0, got {result['score']}"
    assert result["answer_em"] == 1.0
    assert result["has_successful_search"] == 1.0


def test_empty_search_no_reward():
    """Empty search result should not give search reward."""
    sol_failed = (
        "<information>\nSearch failed: No relevant documents were returned.\n</information>\n"
        "<answer>Paris</answer>"
    )
    gt = _make_ground_truth(["Paris"])
    result = compute_score("hotpotqa", sol_failed, gt)
    assert result["has_successful_search"] == 0.0
    # Answer correct but no successful search = 0.70
    assert result["score"] == pytest.approx(0.70)


def test_tool_metrics_override_xml():
    """Structured tool metrics should take priority over XML parsing."""
    solution = (
        "<information>\nSome random info.\n</information>\n"
        "<answer>Paris</answer>"
    )
    gt = _make_ground_truth(["Paris"])

    # XML alone would parse 1 search call
    xml_only = compute_score("hotpotqa", solution, gt)
    assert xml_only["num_search_calls"] == 1

    # But if extra_info says 0 calls, it should override
    extra_info = {"tool/search_called": 0, "tool/search_success": 0, "tool/search_num_docs": 0}
    result = compute_score("hotpotqa", solution, gt, extra_info=extra_info)
    assert result["num_search_calls"] == 0
    assert result["tool_metrics_available"] == 1.0


def test_extract_solution_single_answer():
    """extract_solution() should return the answer when there's exactly one <answer> tag."""
    text = "Some text <answer>Paris</answer> more text"
    ans = extract_solution(text)
    assert ans == "Paris", f"Expected 'Paris', got {ans}"


def test_extract_solution_no_answer():
    """extract_solution() should return None when there's no <answer> tag."""
    text = "Some text without answer tag"
    ans = extract_solution(text)
    assert ans is None


def test_extract_solution_multiple_answers():
    """extract_solution() should return the LAST <answer> tag."""
    text = "<answer>First</answer> <answer>Last</answer>"
    ans = extract_solution(text)
    assert ans == "Last", f"Expected 'Last', got {ans}"


def test_normalize_answer():
    """normalize_answer should handle articles, punctuation, case, and whitespace."""
    assert normalize_answer("Paris") == normalize_answer("paris")
    assert normalize_answer("The capital!") == normalize_answer("capital")
    assert normalize_answer("  extra   spaces  ") == "extra spaces"


def test_em_check():
    """EM check should match against multiple possible answers."""
    assert em_check("Paris", ["Paris", "London"]) == True
    assert em_check("Berlin", ["Paris", "London"]) == False
    assert em_check("paris", ["Paris"]) == True  # case insensitive


if __name__ == "__main__":
    test_hotpot_search_correct()
    test_hotpot_search_correct_no_evidence()
    test_hotpot_no_search_correct()
    test_hotpot_search_failed_correct()
    test_hotpot_search_evidence_wrong_answer()
    test_hotpot_search_wrong_answer()
    test_hotpot_no_search_wrong_answer()
    test_hotpot_no_answer()
    test_nq_no_search_correct()
    test_nq_search_correct()
    test_empty_search_no_reward()
    test_tool_metrics_override_xml()
    test_extract_solution_single_answer()
    test_extract_solution_no_answer()
    test_extract_solution_multiple_answers()
    test_normalize_answer()
    test_em_check()
    print("✅ All reward tests passed!")
