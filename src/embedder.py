from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from .config import EmbeddingConfig


WORD_RE = re.compile(r"[\w\-]+", re.UNICODE)


class HashingEmbedder:
    """Deterministic local embedding backend for reproducible baseline retrieval."""

    def __init__(self, config: EmbeddingConfig):
        self.dimensions = config.dimensions

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for token in WORD_RE.findall(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row_idx, bucket] += sign
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        np.divide(vectors, norms, out=vectors, where=norms > 0)
        return vectors


def embed_chunks(chunks: list[dict], output_path: Path, config: EmbeddingConfig) -> np.ndarray:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embedder = HashingEmbedder(config)
    embeddings = embedder.encode([chunk["text"] for chunk in chunks])
    np.save(output_path, embeddings)
    return embeddings

