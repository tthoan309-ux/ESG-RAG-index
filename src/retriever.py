from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .config import EmbeddingConfig, RetrievalConfig
from .embedder import HashingEmbedder
from .reranker import BGEReranker


TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)


class VectorStore:
    def __init__(self, embeddings: np.ndarray, chunks: list[dict], embedder: HashingEmbedder):
        self.embeddings = embeddings.astype(np.float32)
        self.chunks = chunks
        self.embedder = embedder
        self._bm25 = BM25Index(chunks)

    def search(self, query: str, top_k: int = 5, company: str | None = None, year: int | None = None) -> list[dict]:
        candidate_idx = self._candidate_indexes(company=company, year=year)
        return self._embedding_search(query, candidate_idx, top_k)

    def hybrid_search(
        self,
        query: str,
        top_k: int = RetrievalConfig().top_k,
        prefetch_k: int = RetrievalConfig().prefetch_k,
        company: str | None = None,
        year: int | None = None,
        reranker: BGEReranker | None = None,
    ) -> list[dict]:
        candidate_idx = self._candidate_indexes(company=company, year=year)
        if not candidate_idx:
            return []

        bm25_results = self._bm25.search(query, candidate_idx, prefetch_k)
        embedding_results = self._embedding_search(query, candidate_idx, prefetch_k)
        fused = self._fuse_results(bm25_results, embedding_results, prefetch_k)

        if fused and float(fused[0].get("hybrid_score", 0.0)) >= RetrievalConfig().rerank_threshold:
            for rank, chunk in enumerate(fused[:top_k], start=1):
                chunk["rank"] = rank
                chunk["reranker_model"] = "skipped-high-confidence-hybrid"
                chunk["reranker_score"] = float(chunk.get("hybrid_score", 0.0))
            return fused[:top_k]

        reranker = reranker or BGEReranker()
        reranked = reranker.rerank(query, fused, top_k=top_k)
        for rank, chunk in enumerate(reranked, start=1):
            chunk["rank"] = rank
        return reranked

    def retrieve_for_indicator(
        self,
        indicator: Any,
        top_k: int = RetrievalConfig().top_k,
        company: str | None = None,
        year: int | None = None,
        reranker: BGEReranker | None = None,
    ) -> dict:
        retrieval_query = str(_indicator_value(indicator, "retrieval_query", ""))
        keywords_vi = str(_indicator_value(indicator, "keywords_vi", ""))
        query = f"{retrieval_query} {keywords_vi}".strip()
        retrieved_chunks = self.hybrid_search(
            query,
            top_k=top_k,
            prefetch_k=RetrievalConfig().prefetch_k,
            company=company,
            year=year,
            reranker=reranker,
        )
        return {
            "indicator_id": _indicator_value(indicator, "indicator_id", ""),
            "indicator_name": _indicator_value(indicator, "indicator_name_vi", ""),
            "retrieved_chunks": retrieved_chunks,
        }

    def _candidate_indexes(self, company: str | None, year: int | None) -> list[int]:
        return [
            idx
            for idx, chunk in enumerate(self.chunks)
            if (company is None or chunk["company"] == company) and (year is None or chunk["year"] == year)
        ]

    def _embedding_search(self, query: str, candidate_idx: list[int], top_k: int) -> list[dict]:
        if not candidate_idx:
            return []

        query_vector = self.embedder.encode([query])[0]
        matrix = self.embeddings[candidate_idx]
        scores = matrix @ query_vector
        order = np.argsort(scores)[::-1][:top_k]

        results: list[dict] = []
        for rank, local_idx in enumerate(order, start=1):
            idx = candidate_idx[int(local_idx)]
            item = dict(self.chunks[idx])
            item["embedding_rank"] = rank
            item["embedding_score"] = float(scores[int(local_idx)])
            item["chunk_index"] = idx
            results.append(item)
        return results

    def _fuse_results(self, bm25_results: list[dict], embedding_results: list[dict], prefetch_k: int) -> list[dict]:
        by_key: dict[str, dict] = {}
        for result in bm25_results + embedding_results:
            key = str(result.get("chunk_id") or result.get("chunk_index"))
            existing = by_key.setdefault(key, dict(result))
            existing.update({k: v for k, v in result.items() if k.endswith("_score") or k.endswith("_rank")})

        for result in by_key.values():
            bm25_rank = result.get("bm25_rank", prefetch_k + 1)
            embedding_rank = result.get("embedding_rank", prefetch_k + 1)
            result["hybrid_score"] = (1.0 / (60 + bm25_rank)) + (1.0 / (60 + embedding_rank))

        return sorted(by_key.values(), key=lambda item: item["hybrid_score"], reverse=True)[:prefetch_k]


class BM25Index:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(chunk.get("text", "")) for chunk in chunks]
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.doc_freq: Counter[str] = Counter()
        for doc in self.documents:
            self.doc_freq.update(set(doc))
        self.n_docs = len(self.documents)

    def search(self, query: str, candidate_idx: list[int], top_k: int) -> list[dict]:
        query_terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for idx in candidate_idx:
            score = self._score(query_terms, idx)
            scored.append((score, idx))

        scored.sort(reverse=True, key=lambda item: item[0])
        results: list[dict] = []
        for rank, (score, idx) in enumerate(scored[:top_k], start=1):
            item = dict(self.chunks[idx])
            item["bm25_rank"] = rank
            item["bm25_score"] = float(score)
            item["chunk_index"] = idx
            results.append(item)
        return results

    def _score(self, query_terms: list[str], idx: int) -> float:
        if not query_terms or not self.documents:
            return 0.0

        term_counts = Counter(self.documents[idx])
        doc_len = self.doc_lengths[idx] or 1
        score = 0.0
        for term in query_terms:
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            tf = term_counts.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
            score += idf * numerator / (denominator or 1)
        return score


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_RE.findall(text.lower()):
        normalized = strip_accents(token)
        tokens.append(token)
        if normalized != token:
            tokens.append(normalized)
    return tokens


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _indicator_value(indicator: Any, key: str, default: str) -> Any:
    if isinstance(indicator, dict):
        return indicator.get(key, default)
    return getattr(indicator, key, default)


def save_vectorstore(path: Path, embeddings: np.ndarray, chunks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(chunks, ensure_ascii=False)
    np.savez_compressed(path, embeddings=embeddings.astype(np.float32), chunks=np.array(metadata))


def load_vectorstore(path: Path, config: EmbeddingConfig) -> VectorStore:
    data = np.load(path, allow_pickle=False)
    embeddings = data["embeddings"]
    chunks = json.loads(str(data["chunks"]))
    return VectorStore(embeddings, chunks, HashingEmbedder(config))
