#!/usr/bin/env python3
"""
Unit tests for build_sft_data.py — SFT data building logic.

Tests cover:
    - Hermes tool call parsing with strict validation
    - Role sequence validation
    - Failed search detection
    - Final answer extraction (strict pattern)
    - Token length checking
    - Full filtering pipeline
"""

import json
import os
import sys
import tempfile
import unittest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recipe.search_r1_verl.data.build_sft_data import (
    TOOL_CALL_PATTERN,
    ANSWER_PATTERN,
    SEARCH_FAILURE_MARKERS,
    SEARCH_TOOL_SCHEMA,
    normalize_answer,
    em_check,
    parse_tool_calls,
    validate_tool_calls,
    count_tool_calls,
    validate_role_sequence,
    has_failed_tool_response,
    extract_final_answer,
    has_final_answer,
    check_token_length,
)


class TestParseToolCalls(unittest.TestCase):
    """Tests for Hermes <tool_call> parsing with strict validation."""

    def test_valid_tool_call(self):
        content = (
            '<tool_call>\n'
            '{"name": "search", "arguments": {"query": "test query", "topk": 3}}\n'
            '</tool_call>'
        )
        calls = parse_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "search")
        self.assertEqual(calls[0]["arguments"]["query"], "test query")
        self.assertEqual(calls[0]["arguments"]["topk"], 3)

    def test_multiple_tool_calls(self):
        content = (
            '<tool_call>\n{"name": "search", "arguments": {"query": "q1", "topk": 3}}\n</tool_call>\n'
            'some text\n'
            '<tool_call>\n{"name": "search", "arguments": {"query": "q2", "topk": 5}}\n</tool_call>'
        )
        calls = parse_tool_calls(content)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["arguments"]["query"], "q1")
        self.assertEqual(calls[1]["arguments"]["query"], "q2")

    def test_invalid_json(self):
        content = '<tool_call>\n{invalid json}\n</tool_call>'
        with self.assertRaises(ValueError) as ctx:
            parse_tool_calls(content)
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_wrong_tool_name(self):
        content = (
            '<tool_call>\n'
            '{"name": "wrong_tool", "arguments": {"query": "test"}}\n'
            '</tool_call>'
        )
        with self.assertRaises(ValueError) as ctx:
            parse_tool_calls(content)
        self.assertIn("Tool name must be 'search'", str(ctx.exception))

    def test_empty_query(self):
        content = (
            '<tool_call>\n'
            '{"name": "search", "arguments": {"query": ""}}\n'
            '</tool_call>'
        )
        with self.assertRaises(ValueError) as ctx:
            parse_tool_calls(content)
        self.assertIn("Search query must be a non-empty string", str(ctx.exception))

    def test_query_not_string(self):
        content = (
            '<tool_call>\n'
            '{"name": "search", "arguments": {"query": 123}}\n'
            '</tool_call>'
        )
        with self.assertRaises(ValueError) as ctx:
            parse_tool_calls(content)
        self.assertIn("Search query must be a non-empty string", str(ctx.exception))

    def test_arguments_not_object(self):
        content = (
            '<tool_call>\n'
            '{"name": "search", "arguments": "not an object"}\n'
            '</tool_call>'
        )
        with self.assertRaises(ValueError) as ctx:
            parse_tool_calls(content)
        self.assertIn("Tool arguments must be a JSON object", str(ctx.exception))

    def test_arguemnts_missing(self):
        content = (
            '<tool_call>\n'
            '{"name": "search"}\n'
            '</tool_call>'
        )
        with self.assertRaises(ValueError) as ctx:
            parse_tool_calls(content)
        self.assertIn("Tool arguments must be a JSON object", str(ctx.exception))

    def test_topk_out_of_range(self):
        content = (
            '<tool_call>\n'
            '{"name": "search", "arguments": {"query": "test", "topk": 10}}\n'
            '</tool_call>'
        )
        with self.assertRaises(ValueError) as ctx:
            parse_tool_calls(content)
        self.assertIn("topk must be between 1 and 5", str(ctx.exception))

    def test_topk_not_integer(self):
        content = (
            '<tool_call>\n'
            '{"name": "search", "arguments": {"query": "test", "topk": "3"}}\n'
            '</tool_call>'
        )
        with self.assertRaises(ValueError) as ctx:
            parse_tool_calls(content)
        self.assertIn("topk must be an integer", str(ctx.exception))

    def test_no_tool_calls(self):
        content = "Just some text without tool calls."
        calls = parse_tool_calls(content)
        self.assertEqual(len(calls), 0)

    def test_empty_tool_call_tags(self):
        content = "<tool_call>\n\n</tool_call>"
        calls = parse_tool_calls(content)
        self.assertEqual(len(calls), 0)


class TestValidateToolCalls(unittest.TestCase):
    """Tests for validate_tool_calls function."""

    def test_valid_trajectory(self):
        trajectory = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": (
                '<tool_call>\n{"name": "search", "arguments": {"query": "test"}}\n</tool_call>'
            )},
            {"role": "tool", "content": "<information>result</information>"},
            {"role": "assistant", "content": "<answer>answer</answer>"},
        ]
        valid, reason = validate_tool_calls(trajectory)
        self.assertTrue(valid)
        self.assertEqual(reason, "")

    def test_no_tool_calls(self):
        trajectory = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Direct answer."},
        ]
        valid, reason = validate_tool_calls(trajectory)
        self.assertFalse(valid)
        self.assertEqual(reason, "no_tool_call")

    def test_invalid_tool_call(self):
        trajectory = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": (
                '<tool_call>\n{"name": "wrong", "arguments": {} }\n</tool_call>'
            )},
        ]
        valid, reason = validate_tool_calls(trajectory)
        self.assertFalse(valid)
        self.assertIn("Tool name must be 'search'", reason)


class TestCountToolCalls(unittest.TestCase):
    """Tests for counting tool calls."""

    def test_count_valid_calls(self):
        trajectory = [
            {"role": "assistant", "content": (
                '<tool_call>\n{"name": "search", "arguments": {"query": "q1"}}\n</tool_call>'
            )},
            {"role": "tool", "content": "<information>r1</information>"},
            {"role": "assistant", "content": (
                '<tool_call>\n{"name": "search", "arguments": {"query": "q2"}}\n</tool_call>'
            )},
        ]
        count = count_tool_calls(trajectory)
        self.assertEqual(count, 2)

    def test_no_assistant_turns(self):
        trajectory = [{"role": "user", "content": "hi"}]
        count = count_tool_calls(trajectory)
        self.assertEqual(count, 0)

    def test_no_tool_calls_in_content(self):
        trajectory = [
            {"role": "assistant", "content": "Just thinking."},
        ]
        count = count_tool_calls(trajectory)
        self.assertEqual(count, 0)


class TestValidateRoleSequence(unittest.TestCase):
    """Tests for role sequence validation."""

    def test_valid_sequence(self):
        messages = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "<tool_call>...</tool_call>"},
            {"role": "tool", "content": "<information>result</information>"},
            {"role": "assistant", "content": "<answer>answer</answer>"},
        ]
        valid, reason = validate_role_sequence(messages)
        self.assertTrue(valid)

    def test_missing_user(self):
        messages = [
            {"role": "assistant", "content": "Hello"},
        ]
        valid, reason = validate_role_sequence(messages)
        self.assertFalse(valid)
        self.assertEqual(reason, "missing_user")

    def test_orphan_tool_response(self):
        messages = [
            {"role": "user", "content": "Question?"},
            {"role": "tool", "content": "orphan result"},
        ]
        valid, reason = validate_role_sequence(messages)
        self.assertFalse(valid)
        self.assertEqual(reason, "orphan_tool_response")

    def test_tool_without_tool_call(self):
        messages = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "No tool call here"},
            {"role": "tool", "content": "result"},
        ]
        valid, reason = validate_role_sequence(messages)
        self.assertFalse(valid)
        self.assertEqual(reason, "tool_without_tool_call")

    def test_missing_tool_response(self):
        messages = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "<tool_call>...</tool_call>"},
            {"role": "assistant", "content": "<answer>answer</answer>"},
        ]
        valid, reason = validate_role_sequence(messages)
        self.assertFalse(valid)
        self.assertEqual(reason, "missing_tool_response")

    def test_unknown_role(self):
        messages = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer"},
            {"role": "unknown", "content": "??"},
        ]
        valid, reason = validate_role_sequence(messages)
        self.assertFalse(valid)
        self.assertEqual(reason, "unknown_role: unknown")

    def test_empty_non_system(self):
        messages = [
            {"role": "system", "content": "system"},
        ]
        valid, reason = validate_role_sequence(messages)
        self.assertFalse(valid)
        self.assertEqual(reason, "empty_messages")


class TestHasFailedToolResponse(unittest.TestCase):
    """Tests for detecting failed search responses."""

    def test_successful_response(self):
        trajectory = [
            {"role": "tool", "content": "<information>Doc 1: some content</information>"},
        ]
        self.assertFalse(has_failed_tool_response(trajectory))

    def test_failed_search_marker(self):
        trajectory = [
            {"role": "tool", "content": "<information>Search failed: timeout</information>"},
        ]
        self.assertTrue(has_failed_tool_response(trajectory))

    def test_no_information_tag(self):
        trajectory = [
            {"role": "tool", "content": "Some random content without info tags"},
        ]
        self.assertTrue(has_failed_tool_response(trajectory))

    def test_empty_information(self):
        trajectory = [
            {"role": "tool", "content": "<information>  </information>"},
        ]
        self.assertTrue(has_failed_tool_response(trajectory))

    def test_no_tool_turns(self):
        trajectory = [
            {"role": "assistant", "content": "answer"},
        ]
        self.assertFalse(has_failed_tool_response(trajectory))


class TestExtractFinalAnswer(unittest.TestCase):
    """Tests for extracting final answer from trajectory."""

    def test_valid_answer(self):
        trajectory = [
            {"role": "assistant", "content": "<answer>Paris</answer>"},
        ]
        self.assertEqual(extract_final_answer(trajectory), "Paris")

    def test_answer_with_whitespace(self):
        trajectory = [
            {"role": "assistant", "content": "<answer>  Paris  </answer>"},
        ]
        self.assertEqual(extract_final_answer(trajectory), "Paris")

    def test_last_answer_only(self):
        trajectory = [
            {"role": "assistant", "content": "<answer>first</answer>"},
            {"role": "assistant", "content": "<answer>final</answer>"},
        ]
        self.assertEqual(extract_final_answer(trajectory), "final")

    def test_no_answer(self):
        trajectory = [
            {"role": "assistant", "content": "No answer tag here"},
        ]
        self.assertEqual(extract_final_answer(trajectory), "")

    def test_incomplete_answer_tag(self):
        """Only complete <answer>...</answer> should match."""
        trajectory = [
            {"role": "assistant", "content": "<answer>Beijing"},
        ]
        self.assertEqual(extract_final_answer(trajectory), "")


class TestHasFinalAnswer(unittest.TestCase):
    """Tests for checking if trajectory has final answer."""

    def test_has_complete_answer(self):
        trajectory = [
            {"role": "assistant", "content": "<answer>Paris</answer>"},
        ]
        self.assertTrue(has_final_answer(trajectory))

    def test_no_answer(self):
        trajectory = [
            {"role": "assistant", "content": "Just thinking"},
        ]
        self.assertFalse(has_final_answer(trajectory))

    def test_incomplete_tag(self):
        """Incomplete <answer> tag should NOT pass."""
        trajectory = [
            {"role": "assistant", "content": "<answer>Paris"},
        ]
        self.assertFalse(has_final_answer(trajectory))


class TestNormalizeAnswer(unittest.TestCase):
    """Tests for answer normalization."""

    def test_lowercase(self):
        self.assertEqual(normalize_answer("Paris"), "paris")

    def test_remove_articles(self):
        self.assertEqual(normalize_answer("the capital"), "capital")

    def test_remove_punctuation(self):
        self.assertEqual(normalize_answer("Paris!"), "paris")

    def test_whitespace(self):
        self.assertEqual(normalize_answer("  Paris  "), "paris")


class TestEmCheck(unittest.TestCase):
    """Tests for exact match checking."""

    def test_exact_match(self):
        self.assertTrue(em_check("Paris", ["Paris"]))

    def test_normalized_match(self):
        self.assertTrue(em_check("Paris!", ["paris"]))

    def test_no_match(self):
        self.assertFalse(em_check("London", ["Paris"]))

    def test_multiple_answers(self):
        self.assertTrue(em_check("Paris", ["London", "Paris"]))

    def test_empty_prediction(self):
        self.assertFalse(em_check("", ["Paris"]))


class TestSearchToolSchema(unittest.TestCase):
    """Tests for the search tool schema definition."""

    def test_schema_structure(self):
        self.assertIsInstance(SEARCH_TOOL_SCHEMA, list)
        self.assertEqual(len(SEARCH_TOOL_SCHEMA), 1)
        schema = SEARCH_TOOL_SCHEMA[0]
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "search")
        self.assertIn("query", schema["function"]["parameters"]["required"])


class TestSearchFailureMarkers(unittest.TestCase):
    """Tests for search failure marker constants."""

    def test_markers_exist(self):
        self.assertIn("Search failed:", SEARCH_FAILURE_MARKERS)
        self.assertIn("timed out", SEARCH_FAILURE_MARKERS)
        self.assertIn("HTTP error", SEARCH_FAILURE_MARKERS)
        self.assertIn("Unexpected error", SEARCH_FAILURE_MARKERS)
        self.assertIn("Query is empty", SEARCH_FAILURE_MARKERS)


if __name__ == "__main__":
    unittest.main()
