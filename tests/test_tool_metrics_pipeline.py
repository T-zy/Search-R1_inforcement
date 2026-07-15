"""
Unit tests for the tool metrics pipeline.

Tests that extra_info is correctly parsed by extract_tool_state
and that tool metrics override XML fallback.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recipe.search_r1_verl.rewards.qa_em_tool_reward import (
    extract_tool_state,
    compute_score,
)


def test_metrics_available():
    """When tool/search_called is in extra_info, metrics_available should be True."""
    extra_info = {"tool/search_called": 2, "tool/search_success": 1, "tool/search_failed": 1, "tool/search_num_docs": 3}
    state = extract_tool_state("<information>...</information>", extra_info)
    assert state["metrics_available"] == True
    assert state["num_search_calls"] == 2
    assert state["search_success_count"] == 1
    assert state["search_failed_count"] == 1
    assert state["search_num_docs"] == 3


def test_metrics_not_available():
    """When extra_info is empty or missing tool keys, fall back to XML parsing."""
    state = extract_tool_state("<information>Content</information>", {})
    assert state["metrics_available"] == False
    assert state["num_search_calls"] == 1
    assert state["search_success_count"] == 1


def test_metrics_override_xml():
    """Tool metrics should override what XML parsing would return."""
    # XML says 1 successful search, but metrics say 0 calls
    extra_info = {"tool/search_called": 0, "tool/search_success": 0, "tool/search_num_docs": 0}
    state = extract_tool_state("<information>Content</information>", extra_info)
    assert state["metrics_available"] == True
    assert state["num_search_calls"] == 0
    assert state["has_successful_search"] == False


def test_partial_search_success():
    """Multiple searches, some failing, should be reflected in metrics."""
    extra_info = {
        "tool/search_called": 3,
        "tool/search_success": 2,
        "tool/search_failed": 1,
        "tool/search_num_docs": 5,
    }
    state = extract_tool_state("<information>...</information>", extra_info)
    assert state["has_successful_search"] == True  # at least one success
    assert state["all_searches_successful"] == False  # not all succeeded


def test_all_searches_successful():
    """When all searches succeed, all_searches_successful should be True."""
    extra_info = {
        "tool/search_called": 2,
        "tool/search_success": 2,
        "tool/search_failed": 0,
        "tool/search_num_docs": 4,
    }
    state = extract_tool_state("<information>...</information>", extra_info)
    assert state["all_searches_successful"] == True


def test_timeout_metrics():
    """Timeout counts should be preserved."""
    extra_info = {
        "tool/search_called": 1,
        "tool/search_success": 0,
        "tool/search_failed": 1,
        "tool/search_timeout": 1,
        "tool/search_num_docs": 0,
    }
    state = extract_tool_state("<information>Search failed: timeout</information>", extra_info)
    assert state["search_timeout_count"] == 1
    assert state["has_successful_search"] == False


def test_num_search_calls_in_reward():
    """num_search_calls should appear in reward output."""
    solution = "<information>Content</information><answer>Paris</answer>"
    gt = {"target": ["Paris"]}
    result = compute_score("hotpotqa", solution, gt)
    assert "num_search_calls" in result
    assert result["num_search_calls"] >= 0


def test_tool_metrics_available_in_reward():
    """tool_metrics_available should appear in reward output."""
    solution = "<answer>Paris</answer>"
    gt = {"target": ["Paris"]}
    extra_info = {"tool/search_called": 1, "tool/search_success": 1, "tool/search_num_docs": 1}
    result = compute_score("nq", solution, gt, extra_info=extra_info)
    assert "tool_metrics_available" in result
    assert result["tool_metrics_available"] == 1.0


if __name__ == "__main__":
    test_metrics_available()
    test_metrics_not_available()
    test_metrics_override_xml()
    test_partial_search_success()
    test_all_searches_successful()
    test_timeout_metrics()
    test_num_search_calls_in_reward()
    test_tool_metrics_available_in_reward()
    print("✅✅✅ All tool metrics pipeline tests passed!")
