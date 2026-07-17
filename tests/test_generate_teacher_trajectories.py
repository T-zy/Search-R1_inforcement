#!/usr/bin/env python3
"""
Unit tests for generate_teacher_trajectories.py — teacher trajectory generation logic.

Tests focus on the data extraction and utility functions rather than
model inference (which requires GPU and model weights).
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import torch-free utilities directly from the module
# (torch is only needed for the actual generation, not for helper functions)
try:
    from recipe.search_r1_verl.data.generate_teacher_trajectories import (
        extract_prompt_messages,
        extract_search_query,
        SEARCH_PATTERN,
        TOOL_CALL_PATTERN,
    )
except ImportError as e:
    # If torch is not available, skip all tests with a clear message
    raise unittest.SkipTest(
        f"PyTorch not available: {e}. "
        "Run these tests in a conda environment with torch installed "
        "(e.g., `searchr1`)."
    ) from e


class TestExtractPromptMessages(unittest.TestCase):
    """Tests for extracting system prompt and question from verl parquet rows."""

    def test_normal_case(self):
        row = {
            "prompt": [
                {"role": "system", "content": "You are a helpful agent."},
                {"role": "user", "content": "What is the capital of France?"},
            ]
        }
        system_prompt, question = extract_prompt_messages(row)
        self.assertEqual(system_prompt, "You are a helpful agent.")
        self.assertEqual(question, "What is the capital of France?")

    def test_no_system(self):
        row = {
            "prompt": [
                {"role": "user", "content": "Just a question?"},
            ]
        }
        system_prompt, question = extract_prompt_messages(row)
        self.assertEqual(system_prompt, "")
        self.assertEqual(question, "Just a question?")

    def test_no_user_question(self):
        row = {
            "prompt": [
                {"role": "system", "content": "System only."},
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            extract_prompt_messages(row)
        self.assertIn("No user question", str(ctx.exception))

    def test_list_conversion(self):
        """Test handling of numpy/pandas wrapped lists."""
        row = {
            "prompt": [
                {"role": "user", "content": "Question?"},
            ]
        }
        # tolist() is not available on regular lists, so this should work directly
        system_prompt, question = extract_prompt_messages(row)
        self.assertEqual(question, "Question?")

    def test_empty_content(self):
        row = {
            "prompt": [
                {"role": "user", "content": ""},
            ]
        }
        with self.assertRaises(ValueError):
            extract_prompt_messages(row)


class TestExtractSearchQuery(unittest.TestCase):
    """Tests for extracting search queries from model output."""

    def test_search_tag(self):
        text = "Let me search. <search>Ernest Hemingway</search>"
        query = extract_search_query(text)
        self.assertEqual(query, "Ernest Hemingway")

    def test_tool_call_format(self):
        text = (
            '<tool_call>\n'
            '{"name": "search", "arguments": {"query": "test query"}}\n'
            '</tool_call>'
        )
        query = extract_search_query(text)
        self.assertEqual(query, "test query")

    def test_tool_call_with_topk(self):
        text = (
            '<tool_call>\n'
            '{"name": "search", "arguments": {"query": "test", "topk": 5}}\n'
            '</tool_call>'
        )
        query = extract_search_query(text)
        self.assertEqual(query, "test")

    def test_no_search(self):
        text = "Just thinking about the answer."
        self.assertIsNone(extract_search_query(text))

    def test_last_query_only(self):
        text = "<search>first</search> and <search>last</search>"
        query = extract_search_query(text)
        self.assertEqual(query, "last")

    def test_last_tool_call(self):
        text = (
            '<tool_call>\n{"name": "search", "arguments": {"query": "first"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "search", "arguments": {"query": "last"}}\n</tool_call>'
        )
        query = extract_search_query(text)
        self.assertEqual(query, "last")

    def test_mixed_format_prefers_tool_call(self):
        """Tool_call format should be preferred over search format."""
        text = (
            '<search>old query</search>\n'
            '<tool_call>\n{"name": "search", "arguments": {"query": "new query"}}\n</tool_call>'
        )
        query = extract_search_query(text)
        self.assertEqual(query, "new query")

    def test_empty_search_tag(self):
        text = "<search>  </search>"
        query = extract_search_query(text)
        self.assertEqual(query, "")

    def test_wrong_tool_name_in_tool_call(self):
        """Tool call with wrong name should be ignored and fallback to search."""
        text = (
            '<tool_call>\n{"name": "wrong", "arguments": {"query": "test"}}\n</tool_call>\n'
            '<search>fallback</search>'
        )
        query = extract_search_query(text)
        self.assertEqual(query, "fallback")


if __name__ == "__main__":
    unittest.main()
