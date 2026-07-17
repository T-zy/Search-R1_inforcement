#!/usr/bin/env python3
"""
Unit tests for convert_to_hermes_sft.py — old <search> to Hermes <tool_call> conversion.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recipe.search_r1_verl.data.convert_to_hermes_sft import (
    convert_search_to_tool_call,
    is_already_hermes,
    convert_trajectory,
)


class TestIsAlreadyHermes(unittest.TestCase):
    """Tests for detecting if text already uses Hermes format."""

    def test_hermes_format(self):
        self.assertTrue(is_already_hermes("<tool_call>...</tool_call>"))

    def test_old_format(self):
        self.assertFalse(is_already_hermes("<search>query</search>"))

    def test_no_tags(self):
        self.assertFalse(is_already_hermes("Just text"))

    def test_mixed(self):
        self.assertTrue(is_already_hermes("text <tool_call>...</tool_call> text"))


class TestConvertSearchToToolCall(unittest.TestCase):
    """Tests for converting <search> to <tool_call> format."""

    def test_raw_text_query(self):
        result = convert_search_to_tool_call("<search>Ernest Hemingway</search>")
        self.assertIn("<tool_call>", result)
        self.assertIn("Ernest Hemingway", result)
        self.assertIn("name", result)
        self.assertIn("search", result)
        # Verify it's valid JSON inside the tool_call
        payload = json.loads(result.split("<tool_call>")[1].split("</tool_call>")[0].strip())
        self.assertEqual(payload["name"], "search")
        self.assertEqual(payload["arguments"]["query"], "Ernest Hemingway")

    def test_json_query_object(self):
        """Query that is already a JSON object should be wrapped as arguments."""
        result = convert_search_to_tool_call('<search>{"query": "test", "topk": 5}</search>')
        payload = json.loads(result.split("<tool_call>")[1].split("</tool_call>")[0].strip())
        self.assertEqual(payload["name"], "search")
        self.assertEqual(payload["arguments"]["query"], "test")
        self.assertEqual(payload["arguments"]["topk"], 5)

    def test_json_with_name_and_arguments(self):
        """Query that is already in full {name, arguments} format should pass through."""
        result = convert_search_to_tool_call(
            '<search>{"name": "search", "arguments": {"query": "test"}}</search>'
        )
        payload = json.loads(result.split("<tool_call>")[1].split("</tool_call>")[0].strip())
        self.assertEqual(payload["name"], "search")
        self.assertEqual(payload["arguments"]["query"], "test")

    def test_no_search_tag(self):
        result = convert_search_to_tool_call("No search tag here")
        self.assertEqual(result, "No search tag here")

    def test_empty_search(self):
        result = convert_search_to_tool_call("<search>  </search>")
        payload = json.loads(result.split("<tool_call>")[1].split("</tool_call>")[0].strip())
        self.assertEqual(payload["arguments"]["query"], "")

    def test_multiple_searches(self):
        result = convert_search_to_tool_call(
            "<search>first query</search> and <search>second query</search>"
        )
        # Should have two tool calls
        self.assertEqual(result.count("<tool_call>"), 2)


class TestConvertTrajectory(unittest.TestCase):
    """Tests for converting full trajectories."""

    def test_convert_old_format(self):
        trajectory = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Let me search. <search>my query</search>"},
            {"role": "tool", "content": "<information>result</information>"},
            {"role": "assistant", "content": "<answer>answer</answer>"},
        ]
        new_traj = convert_trajectory(trajectory)
        self.assertIn("<tool_call>", new_traj[1]["content"])
        self.assertNotIn("<search>", new_traj[1]["content"])
        # Non-assistant messages should be unchanged
        self.assertEqual(new_traj[0], trajectory[0])
        self.assertEqual(new_traj[2], trajectory[2])

    def test_already_hermes_idempotent(self):
        trajectory = [
            {"role": "assistant", "content": (
                '<tool_call>\n{"name": "search", "arguments": {"query": "test"}}\n</tool_call>'
            )},
        ]
        new_traj = convert_trajectory(trajectory)
        self.assertEqual(new_traj[0], trajectory[0])

    def test_empty_trajectory(self):
        self.assertEqual(convert_trajectory([]), [])


if __name__ == "__main__":
    unittest.main()
