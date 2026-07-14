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
Search tool for verl native tool agent loop.
Wraps the local E5 + FAISS IVF retrieval service as a verl BaseTool.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional
from functools import lru_cache

import aiohttp

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SearchTool(BaseTool):
    """
    Native verl tool that calls a local retrieval service via HTTP.

    The retrieval service is expected to be a FastAPI server running at
    ``endpoint`` (default http://127.0.0.1:8000/retrieve) with the same
    request/response schema as ``search_r1/search/retrieval_server.py``.

    Tool parameters (from LLM):
        query (str): The search query.
        topk (int, optional): Number of passages to retrieve. Default is from config.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        # Provide a default schema so the YAML config can omit tool_schema
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate({
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search Wikipedia passages relevant to a factual question. "
                                   "Returns the top-k passages formatted as <information>...</information>.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string."
                            },
                            "topk": {
                                "type": "integer",
                                "description": "Number of passages to retrieve (1-5). Default is 3."
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        super().__init__(config, tool_schema)

        self.endpoint = config.get("endpoint", "http://127.0.0.1:8000/retrieve")
        self.timeout = config.get("timeout", 5)
        self.default_topk = config.get("default_topk", 3)
        self.max_topk = config.get("max_topk", 5)
        self.max_doc_chars = config.get("max_doc_chars", 1200)
        self.max_tool_response_chars = config.get("max_tool_response_chars", 6000)

        # Per-instance metrics (keyed by instance_id for concurrency safety)
        self._metrics_by_instance: dict[str, dict] = {}
        self._latencies_by_instance: dict[str, list] = {}
        # Lazily-initialised HTTP session (reused across calls)
        self._http_session: aiohttp.ClientSession | None = None

    async def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._http_session

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Reset per-trajectory metrics."""
        instance_id, response = await super().create(instance_id, **kwargs)
        self._metrics_by_instance[instance_id] = {}
        self._latencies_by_instance[instance_id] = []
        return instance_id, response

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        """
        Execute a search query against the local retrieval service.

        Returns:
            ToolResponse with formatted <information> block.
            reward (always 0.0 — rewards are computed separately by RewardManager).
            metrics dict with timing and error info.
        """
        query = parameters.get("query", "").strip()
        topk = parameters.get("topk", self.default_topk)

        # --- Input validation ---
        if not query:
            return self._make_error_response("empty_query", "Query is empty.")

        topk = min(max(1, int(topk) if topk else self.default_topk), self.max_topk)

        # --- Call retrieval service ---
        start_time = time.monotonic()
        try:
            response_text, error_type = await self._call_retrieval_service(query, topk)
        except asyncio.TimeoutError:
            latency_ms = (time.monotonic() - start_time) * 1000
            self._record_metrics(instance_id, "timeout", latency_ms)
            return self._make_error_response(
                "timeout",
                f"Retrieval service timed out after {self.timeout}s.",
                latency_ms=latency_ms,
            )
        except aiohttp.ClientError as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            self._record_metrics(instance_id, "http_error", latency_ms)
            return self._make_error_response(
                "http_error",
                f"HTTP error calling retrieval service: {e}",
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            self._record_metrics(instance_id, "unknown_error", latency_ms)
            logger.exception("SearchTool: unexpected error during retrieval")
            return self._make_error_response(
                "unknown_error",
                f"Unexpected error: {e}",
                latency_ms=latency_ms,
            )

        latency_ms = (time.monotonic() - start_time) * 1000
        self._record_metrics(instance_id, "success", latency_ms)

        # --- Format tool response ---
        num_docs = 0
        response_truncated = False

        try:
            parsed = json.loads(response_text)
            results = parsed.get("result", [])
            if results and isinstance(results, list):
                docs = results[0]  # single-query batch
            else:
                docs = []
        except (json.JSONDecodeError, IndexError, TypeError):
            docs = []

        formatted_parts = []
        for i, doc in enumerate(docs):
            if isinstance(doc, dict):
                # Support both nested {"document": {...}} and flat doc formats
                doc_content = doc.get("document", doc)
                title = doc_content.get("title", "")
                text = doc_content.get("text", "") or doc_content.get("contents", "")
                if not text:
                    text = str(doc_content)
            else:
                title = ""
                text = str(doc)

            # Truncate individual document
            if len(text) > self.max_doc_chars:
                text = text[:self.max_doc_chars] + "..."
                response_truncated = True

            part = f"Doc {i + 1}"
            if title:
                part += f"(Title: {title})"
            part += f":\n{text}"
            formatted_parts.append(part)
            num_docs += 1

        formatted_text = "<information>\n" + "\n\n".join(formatted_parts) + "\n</information>"

        # Truncate entire tool response if needed
        if len(formatted_text) > self.max_tool_response_chars:
            formatted_text = formatted_text[:self.max_tool_response_chars] + "\n... [truncated]"
            response_truncated = True

        tool_response = ToolResponse(text=formatted_text)

        metrics = {
            "tool/search_called": 1,
            "tool/search_success": 1,
            "tool/search_failed": 0,
            "tool/search_timeout": 0,
            "tool/search_empty_query": 0,
            "tool/search_latency_ms": latency_ms,
            "tool/search_num_docs": num_docs,
            "tool/search_response_truncated": int(response_truncated),
            "tool/search_exception_type": "none",
        }

        return tool_response, 0.0, metrics

    async def _call_retrieval_service(self, query: str, topk: int) -> tuple[str, str]:
        """Make HTTP request to the local retrieval service (reuses session)."""
        session = await self._get_http_session()
        payload = {
            "queries": [query],
            "topk": topk,
            "return_scores": False,
            "max_doc_chars": self.max_doc_chars,
        }
        async with session.post(
            self.endpoint,
            json=payload,
        ) as resp:
            resp.raise_for_status()
            text = await resp.text()
            return text, "none"

    def _make_error_response(
        self,
        error_type: str,
        message: str,
        latency_ms: float = 0.0,
    ) -> tuple[ToolResponse, float, dict]:
        """Create a tool response for error cases."""
        error_text = f"<information>\nSearch failed: {message}\n</information>"
        tool_response = ToolResponse(text=error_text)

        metrics = {
            "tool/search_called": 1,
            "tool/search_success": 0,
            "tool/search_failed": 1,
            "tool/search_timeout": int(error_type == "timeout"),
            "tool/search_empty_query": int(error_type == "empty_query"),
            "tool/search_latency_ms": latency_ms,
            "tool/search_num_docs": 0,
            "tool/search_response_truncated": 0,
            "tool/search_exception_type": error_type,
        }

        return tool_response, 0.0, metrics

    def _record_metrics(self, instance_id: str, status: str, latency_ms: float) -> None:
        """Update per-instance metrics."""
        if instance_id not in self._metrics_by_instance:
            self._metrics_by_instance[instance_id] = {}
            self._latencies_by_instance[instance_id] = []
        self._metrics_by_instance[instance_id][status] = \
            self._metrics_by_instance[instance_id].get(status, 0) + 1
        self._latencies_by_instance[instance_id].append(latency_ms)

    async def release(self, instance_id: str, **kwargs) -> None:
        """Log per-instance metrics and close HTTP session."""
        latencies = self._latencies_by_instance.get(instance_id, [])
        metrics = self._metrics_by_instance.get(instance_id, {})
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            logger.info(
                "SearchTool [%s] stats: calls=%d, success=%d, errors=%d, avg_latency=%.1fms",
                instance_id[:8],
                sum(metrics.values()),
                metrics.get("success", 0),
                metrics.get("http_error", 0) + metrics.get("timeout", 0),
                avg_latency,
            )
        # Cleanup per-instance state
        self._metrics_by_instance.pop(instance_id, None)
        self._latencies_by_instance.pop(instance_id, None)
        await super().release(instance_id, **kwargs)
