from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .cache_manager import CacheManager
from .config import EmbeddingConfig
from .embedder import HashingEmbedder
from .manifest_manager import ManifestManager
from .progress import progress


@dataclass
class EmbeddingResult:
    embeddings: np.ndarray
    embedding_count: int
    reused_count: int
    new_count: int


class EmbeddingManager:
    def __init__(self, cache: CacheManager, manifest: ManifestManager, config: EmbeddingConfig, batch_size: int = 5000):
        self.cache = cache
        self.manifest = manifest
        self.config = config
        self.batch_size = batch_size
        self.embedder = HashingEmbedder(config)

    def embed(self, chunks: list[dict], rebuild: bool = False) -> EmbeddingResult:
        grouped = self._group_by_report(chunks)
        arrays: list[np.ndarray] = []
        reused = 0
        new = 0
        report_embeddings: dict[str, str] = {}

        for report_id, report_chunks in progress(grouped.items(), desc="Embedding", unit="report"):
            identity = self._identity_from_report_chunks(report_id)
            path = self.cache.embedding_path(report_id)
            expected_count = len(report_chunks)
            if not rebuild and path.exists() and identity and self.manifest.is_complete(identity, "embedded"):
                arr = np.load(path)
                if len(arr) == expected_count:
                    arrays.append(arr)
                    reused += expected_count
                    report_embeddings[report_id] = str(path)
                    continue

            arr = self._embed_report(report_chunks)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, arr)
            if identity:
                self.manifest.mark(identity, "embedded")
            arrays.append(arr)
            new += expected_count
            report_embeddings[report_id] = str(path)

        embeddings = np.vstack(arrays) if arrays else np.empty((0, self.config.dimensions), dtype=np.float32)
        self.cache.save_embeddings(embeddings)
        self.cache.write_embedding_manifest([str(chunk["chunk_id"]) for chunk in chunks], report_embeddings)
        self.manifest.save()
        return EmbeddingResult(embeddings, len(chunks), reused, new)

    def _embed_report(self, chunks: list[dict]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            batches.append(self.embedder.encode([str(chunk["text"]) for chunk in batch]))
        return np.vstack(batches) if batches else np.empty((0, self.config.dimensions), dtype=np.float32)

    def _group_by_report(self, chunks: list[dict]) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            source = str(chunk.get("source_file", "unknown"))
            report_id = source.rsplit(".", 1)[0]
            grouped[report_id].append(chunk)
        return dict(grouped)

    def _identity_from_report_chunks(self, report_id: str):
        row = self.manifest.frame.loc[self.manifest.frame["report_id"] == report_id]
        if row.empty:
            return None
        from .manifest_manager import ReportIdentity
        from pathlib import Path

        first = row.iloc[0]
        return ReportIdentity(report_id=str(first["report_id"]), file_hash=str(first["file_hash"]), path=Path(""))
