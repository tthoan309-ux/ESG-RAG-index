from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd


WORD_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(str(text)) if len(token) > 1]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.tokens = [tokenize(document) for document in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.k1, self.b = k1, b
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        count = len(self.tokens)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.tokens), dtype=float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term, 0.0)
            for index, frequencies in enumerate(self.term_frequencies):
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                norm = frequency + self.k1 * (
                    1 - self.b + self.b * self.lengths[index] / max(self.avg_length, 1)
                )
                scores[index] += idf * frequency * (self.k1 + 1) / norm
        return scores


@lru_cache(maxsize=2)
def _load_dense_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Hybrid retrieval requires sentence-transformers. Install requirements-hybrid.txt."
        ) from exc
    return SentenceTransformer(model_name)


def _rank_map(scores: np.ndarray, eligible: np.ndarray, top_k: int) -> dict[int, int]:
    ordered = [int(index) for index in np.argsort(-scores) if eligible[index]][:top_k]
    return {index: rank for rank, index in enumerate(ordered, start=1)}


def retrieve_candidates(
    chunks: pd.DataFrame,
    codebook: dict[str, Any],
    *,
    mode: str = "bm25",
    top_k: int = 6,
    min_bm25_score: float = 0.05,
    dense_model: str = "BAAI/bge-m3",
    min_dense_score: float = 0.25,
    rrf_k: int = 60,
) -> pd.DataFrame:
    if mode not in {"bm25", "hybrid"}:
        raise ValueError("retrieval mode must be bm25 or hybrid.")
    rows: list[dict] = []
    for (ticker, year), group in chunks.groupby(["ticker", "year"], sort=True):
        group = group.reset_index(drop=True)
        texts = group["text"].tolist()
        lexical = BM25(texts)
        dense_model_instance = _load_dense_model(dense_model) if mode == "hybrid" else None
        dense_document_vectors = (
            np.asarray(dense_model_instance.encode(texts, normalize_embeddings=True, show_progress_bar=False))
            if dense_model_instance is not None else None
        )
        for indicator in codebook["indicators"]:
            query = " ; ".join(map(str, indicator["retrieval_queries"]))
            bm25_scores = lexical.score(query)
            bm25_ranks = _rank_map(bm25_scores, bm25_scores >= min_bm25_score, top_k)
            dense_scores = np.zeros(len(group), dtype=float)
            dense_ranks: dict[int, int] = {}
            if dense_model_instance is not None and len(group):
                query_vector = np.asarray(dense_model_instance.encode(
                    query, normalize_embeddings=True, show_progress_bar=False,
                ))
                dense_scores = dense_document_vectors @ query_vector
                dense_ranks = _rank_map(dense_scores, dense_scores >= min_dense_score, top_k)
            candidates = sorted(set(bm25_ranks) | set(dense_ranks))
            scored = []
            for index in candidates:
                fused = (1 / (rrf_k + bm25_ranks[index]) if index in bm25_ranks else 0.0)
                fused += (1 / (rrf_k + dense_ranks[index]) if index in dense_ranks else 0.0)
                scored.append((index, fused))
            for candidate_rank, (index, fused) in enumerate(sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k], start=1):
                source = group.iloc[index]
                rows.append({
                    "ticker": ticker,
                    "year": year,
                    "indicator_id": indicator["indicator_id"],
                    "candidate_rank": candidate_rank,
                    "chunk_id": source["chunk_id"],
                    "source_document_id": source["source_document_id"],
                    "page_number": source.get("page_number"),
                    "provenance_status": source["provenance_status"],
                    "bm25_score": float(bm25_scores[index]),
                    "dense_score": float(dense_scores[index]) if mode == "hybrid" else pd.NA,
                    "fusion_score": float(fused),
                    "text": source["text"],
                })
    return pd.DataFrame(rows, columns=[
        "ticker", "year", "indicator_id", "candidate_rank", "chunk_id",
        "source_document_id", "page_number", "provenance_status", "bm25_score",
        "dense_score", "fusion_score", "text",
    ])
