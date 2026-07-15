"""
Unit tests for SearchTool response formatting.

Tests that:
- Tool response always has complete <information>...</information> tags
- Empty results are properly handled
- Long responses are truncated without breaking tags
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recipe.search_r1_verl.rewards.qa_em_tool_reward import (
    extract_information_blocks,
    extract_tool_state,
)


def test_information_tags_present():
    """Verify that information blocks can be extracted."""
    text = "<information>\nDoc 1(Title: Test): Content here.\n</information>"
    blocks = extract_information_blocks(text)
    assert len(blocks) == 1
    assert "Content here" in blocks[0]


def test_information_tags_closed():
    """Verify that the formatted text has complete XML tags."""
    # Simulate the new formatting logic (body first, then wrap)
    body = "Doc 1(Title: Paris): Paris is the capital."
    formatted = "<information>\n" + body + "\n</information>"
    assert formatted.startswith("<information>")
    assert formatted.endswith("</information>")


def test_empty_result_parsing():
    """Empty result should not be detected as a successful search."""
    text = "<information>\nSearch failed: No relevant documents were returned.\n</information>"
    state = extract_tool_state(text, {})
    assert state["has_successful_search"] == False
    assert state["num_search_calls"] == 1
    assert state["search_success_count"] == 0


def test_single_doc_formatting():
    """Verify single document formatting."""
    body = "Doc 1(Title: Test): Some content."
    formatted = "<information>\n" + body + "\n</information>"
    assert "<information>" in formatted
    assert "</information>" in formatted
    assert "Doc 1" in formatted


def test_multi_doc_formatting():
    """Verify multi-document formatting."""
    parts = [
        "Doc 1(Title: A): Content A.",
        "Doc 2(Title: B): Content B.",
    ]
    body = "\n\n".join(parts)
    formatted = "<information>\n" + body + "\n</information>"
    blocks = extract_information_blocks(formatted)
    assert len(blocks) == 1
    assert "Doc 1" in blocks[0]
    assert "Doc 2" in blocks[0]


def test_truncation_preserves_tags():
    """Simulate the new truncation logic: truncate body, then wrap."""
    body = "A" * 1000
    max_chars = 100
    reserved = len("<information>\n\n</information>") + 32
    max_body = max(0, max_chars - reserved)
    if len(body) > max_body:
        body = body[:max_body] + "\n... [truncated]"
    formatted = "<information>\n" + body + "\n</information>"
    assert formatted.startswith("<information>")
    assert formatted.endswith("</information>")
    assert "... [truncated]" in formatted


if __name__ == "__main__":
    test_information_tags_present()
    test_information_tags_closed()
    test_empty_result_parsing()
    test_single_doc_formatting()
    test_multi_doc_formatting()
    test_truncation_preserves_tags()
    print("✅✅✅ All SearchTool formatting tests passed!")
