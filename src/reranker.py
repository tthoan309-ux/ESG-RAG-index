from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock

from .config import RetrievalConfig


LOGGER = logging.getLogger(__name__)


@dataclass
class BGEReranker:
    model_name: str = RetrievalConfig().reranker_model

    def __post_init__(self) -> None:
        self._model = None
        self._lock = Lock()

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        if not chunks:
            return []

        model = self._load_model()
        if model is None:
            LOGGER.warning("Reranker unavailable; falling back to hybrid scores.")
            ranked = sorted(chunks, key=lambda item: item.get("hybrid_score", 0.0), reverse=True)
            for item in ranked:
                item["reranker_model"] = "fallback-hybrid-score"
                item["reranker_score"] = float(item.get("hybrid_score", 0.0))
            return ranked[:top_k]

        pairs = [(query, str(chunk.get("text", ""))) for chunk in chunks]
        scores = model.predict(pairs)
        ranked: list[dict] = []
        for chunk, score in zip(chunks, scores):
            item = dict(chunk)
            item["reranker_model"] = self.model_name
            item["reranker_score"] = float(score)
            ranked.append(item)
        return sorted(ranked, key=lambda item: item["reranker_score"], reverse=True)[:top_k]

    def rerank_many(self, jobs: list[tuple[str, list[dict], int]]) -> list[list[dict]]:
        if not jobs:
            return []

        model = self._load_model()
        if model is None:
            results: list[list[dict]] = []
            for _, chunks, top_k in jobs:
                ranked = sorted(chunks, key=lambda item: item.get("retrieval_score", item.get("hybrid_score", 0.0)), reverse=True)
                for item in ranked:
                    item["reranker_model"] = "fallback-retrieval-score"
                    item["reranker_score"] = float(item.get("retrieval_score", item.get("hybrid_score", 0.0)))
                results.append(ranked[:top_k])
            return results

        pairs: list[tuple[str, str]] = []
        spans: list[tuple[int, int, int]] = []
        cursor = 0
        for query, chunks, top_k in jobs:
            start = cursor
            for chunk in chunks:
                pairs.append((query, str(chunk.get("text", ""))))
                cursor += 1
            spans.append((start, cursor, top_k))

        with self._lock:
            scores = model.predict(pairs) if pairs else []
        outputs: list[list[dict]] = []
        for (start, end, top_k), (_, chunks, _) in zip(spans, jobs):
            ranked: list[dict] = []
            for chunk, score in zip(chunks, scores[start:end]):
                item = dict(chunk)
                item["reranker_model"] = self.model_name
                item["reranker_score"] = float(score)
                ranked.append(item)
            outputs.append(sorted(ranked, key=lambda item: item["reranker_score"], reverse=True)[:top_k])
        return outputs

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            return self._model
        except Exception as exc:
            LOGGER.warning("Could not load reranker model %s: %s", self.model_name, exc)
            return None
