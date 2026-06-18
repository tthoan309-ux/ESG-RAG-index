from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import ChunkingConfig, EmbeddingConfig, RetrievalConfig


def write_experiment(
    output_dir: Path,
    embedding_config: EmbeddingConfig,
    retrieval_config: RetrievalConfig,
    chunk_config: ChunkingConfig,
    runtime_seconds: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    experiment_id = timestamp.replace(":", "").replace(".", "").replace("+", "Z")
    row = {
        "experiment_id": experiment_id,
        "embedding_model": f"hashing-{embedding_config.dimensions}",
        "reranker_model": retrieval_config.reranker_model,
        "chunk_size": chunk_config.chunk_size,
        "top_k": retrieval_config.top_k,
        "retrieval_mode": retrieval_config.retrieval_mode,
        "rerank_threshold": retrieval_config.rerank_threshold,
        "warehouse_top_k": retrieval_config.warehouse_top_k,
        "runtime": runtime_seconds,
        "timestamp": timestamp,
    }
    pd.DataFrame([row]).to_csv(output_dir / f"{experiment_id}.csv", index=False)
    (output_dir / f"{experiment_id}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row
