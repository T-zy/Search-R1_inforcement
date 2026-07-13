#!/usr/bin/env python3
"""
Enhanced retrieval service for Search-R1.

Retains the proven E5 + FAISS IVF + FastAPI pipeline from the original
Search-R1, with added:

- ``/health`` endpoint for monitoring
- Rich ``/retrieve`` request parameters (max_doc_chars, return_scores)
- Server-side protection (topk cap, empty query, truncation)
- LRU query cache
- Latency tracking
- Comprehensive error handling

Usage:
    python server.py \\
        --index_path /path/to/e5_IVF4096_Flat.index \\
        --corpus_path /path/to/wiki-18.jsonl \\
        --retriever_model intfloat/e5-base-v2 \\
        --port 8000
"""

import argparse
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("retrieval_service")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    index_loaded: bool = False
    corpus_loaded: bool = False
    retriever: str = ""
    index_type: str = ""
    topk_default: int = 3
    total_docs: int = 0


class RetrieveRequest(BaseModel):
    queries: list[str] = Field(default_factory=list, description="List of search queries.")
    topk: Optional[int] = Field(default=None, ge=1, le=5, description="Number of passages per query.")
    return_scores: bool = Field(default=False, description="Whether to return similarity scores.")
    max_doc_chars: int = Field(default=1200, ge=100, le=10000, description="Max chars per document.")


class RetrieveResponse(BaseModel):
    result: list[list[dict[str, Any]]] = Field(default_factory=list, description="Retrieved passages per query.")
    latencies_ms: list[float] = Field(default_factory=list, description="Latency per query in ms.")


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------

class LRUCache:
    """Simple LRU cache for search results."""

    def __init__(self, capacity: int = 1000):
        self.cache: OrderedDict = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def pooling(
    pooler_output: torch.Tensor,
    last_hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pooling_method: str = "mean",
) -> torch.Tensor:
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError(f"Pooling method '{pooling_method}' not implemented.")


class Encoder:
    """Encoder wrapper for embedding queries."""

    def __init__(self, model_name: str, model_path: str, pooling_method: str = "mean",
                 max_length: int = 256, use_fp16: bool = True):
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.max_length = max_length

        logger.info(f"Loading encoder model from {model_path} ...")
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        if use_fp16 and torch.cuda.is_available():
            self.model = self.model.half()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        logger.info(f"Encoder loaded. CUDA available: {torch.cuda.is_available()}")

    @torch.no_grad()
    def encode(self, query_list: list[str], is_query: bool = True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        # Apply E5 prefix
        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {q}" for q in query_list]
            else:
                query_list = [f"passage: {q}" for q in query_list]

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        output = self.model(**inputs, return_dict=True)
        query_emb = pooling(
            output.pooler_output,
            output.last_hidden_state,
            inputs["attention_mask"],
            self.pooling_method,
        )
        if "dpr" not in self.model_name.lower():
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy().astype(np.float32, order="C")
        return query_emb


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: str):
    """Load corpus from a JSONL file using datasets."""
    logger.info(f"Loading corpus from {corpus_path} ...")
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    logger.info(f"Corpus loaded: {len(corpus)} documents.")
    return corpus


def load_docs(corpus, doc_idxs: list[int]) -> list[dict]:
    """Load documents from corpus by indices."""
    return [corpus[int(idx)] for idx in doc_idxs]


class DenseRetriever:
    """Dense retriever using FAISS IVF index."""

    def __init__(self, index_path: str, corpus_path: str, model_name: str,
                 model_path: str, topk: int = 3, use_fp16: bool = True):
        self.topk = topk

        # Load FAISS index
        logger.info(f"Loading FAISS index from {index_path} ...")
        self.index = faiss.read_index(index_path)
        logger.info(f"FAISS index loaded. Dimension: {self.index.d}, "
                    f"n_total: {self.index.ntotal}")

        # Determine index type
        index_type_str = type(self.index).__name__
        if hasattr(self.index, 'invlists'):
            index_type_str += f"_IVF{self.index.invlists.nlist}"

        self.index_type = index_type_str

        # Load corpus
        self.corpus = load_corpus(corpus_path)

        # Load encoder
        self.encoder = Encoder(
            model_name=model_name,
            model_path=model_path,
            pooling_method="mean",
            max_length=256,
            use_fp16=use_fp16,
        )

        # LRU cache
        self.cache = LRUCache(capacity=2000)

    def search(self, query: str, num: int = None, return_score: bool = False,
               max_doc_chars: int = 1200) -> tuple[list[dict], Optional[list[float]]]:
        """Search for a single query."""
        if num is None:
            num = self.topk

        # Check cache
        cache_key = f"{query}:{num}:{max_doc_chars}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Encode query
        query_emb = self.encoder.encode(query)

        # Search index
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]

        # Load documents
        results = load_docs(self.corpus, idxs)

        # Format results
        formatted = []
        for doc, score in zip(results, scores):
            entry = {
                "title": doc.get("title", ""),
                "text": doc.get("text", "") or doc.get("contents", ""),
                "contents": doc.get("contents", str(doc)),
            }
            # Truncate text
            if len(entry["text"]) > max_doc_chars:
                entry["text"] = entry["text"][:max_doc_chars] + "..."
            if return_score:
                entry["score"] = float(score)
            formatted.append(entry)

        output = (formatted, scores.tolist() if return_score else None)
        self.cache.put(cache_key, output)
        return output

    @property
    def total_docs(self) -> int:
        return len(self.corpus)


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(title="Search-R1 Retrieval Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever: Optional[DenseRetriever] = None


@app.on_event("startup")
async def startup_event():
    global retriever
    logger.info("Retrieval service started.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if retriever is None:
        return HealthResponse(
            status="error",
            index_loaded=False,
            corpus_loaded=False,
        )
    return HealthResponse(
        status="ok",
        index_loaded=retriever.index is not None,
        corpus_loaded=retriever.corpus is not None,
        retriever=retriever.encoder.model_name,
        index_type=retriever.index_type,
        topk_default=retriever.topk,
        total_docs=retriever.total_docs,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(request: RetrieveRequest, http_request: Request):
    """Retrieve passages for a list of queries."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized.")

    queries = request.queries
    topk = request.topk or retriever.topk
    return_scores = request.return_scores
    max_doc_chars = request.max_doc_chars

    # Guard: empty queries
    if not queries:
        return RetrieveResponse(result=[], latencies_ms=[])

    results = []
    latencies = []

    for query in queries:
        # Guard: empty query string
        if not query.strip():
            results.append([])
            latencies.append(0.0)
            continue

        start = time.monotonic()
        try:
            docs, scores = retriever.search(
                query=query.strip(),
                num=topk,
                return_score=return_scores,
                max_doc_chars=max_doc_chars,
            )
            latency = (time.monotonic() - start) * 1000

            if return_scores:
                formatted_docs = []
                for doc, score in zip(docs, scores):
                    formatted_docs.append({"document": doc, "score": score})
                results.append(formatted_docs)
            else:
                results.append(docs)

            latencies.append(latency)
        except Exception as e:
            logger.error(f"Search failed for query '{query[:50]}': {e}")
            results.append([])
            latencies.append(0.0)

    return RetrieveResponse(result=results, latencies_ms=latencies)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced retrieval service for Search-R1")
    parser.add_argument("--index_path", type=str, required=True,
                        help="Path to FAISS index file (e.g., e5_IVF4096_Flat.index)")
    parser.add_argument("--corpus_path", type=str, required=True,
                        help="Path to corpus JSONL file (e.g., wiki-18.jsonl)")
    parser.add_argument("--retriever_name", type=str, default="e5",
                        help="Name of the retriever model")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2",
                        help="Path or name of the retriever model")
    parser.add_argument("--topk", type=int, default=3,
                        help="Default number of passages to retrieve")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--use_fp16", action="store_true", default=True,
                        help="Use FP16 for encoder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialize retriever globally
    retriever = DenseRetriever(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        model_name=args.retriever_name,
        model_path=args.retriever_model,
        topk=args.topk,
        use_fp16=args.use_fp16,
    )
    logger.info("Retriever initialized. Starting server...")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
