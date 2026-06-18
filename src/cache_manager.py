from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .storage import read_table, write_table


@dataclass(frozen=True)
class ReportCacheRecord:
    report_id: str
    source_file: str
    file_hash: str
    parsed_path: str
    chunk_path: str
    embedding_path: str
    used_ocr: bool
    page_count: int
    text_length: int
    chunk_count: int


class CacheManager:
    def __init__(self, root: Path):
        self.root = root
        self.parsed_dir = root / "data" / "parsed_reports"
        self.chunks_dir = root / "data" / "chunks"
        self.embeddings_dir = root / "data" / "embeddings"
        self.outputs_dir = root / "outputs"
        self.report_manifest_path = self.outputs_dir / "report_cache_manifest.json"
        self.embedding_manifest_path = self.embeddings_dir / "embedding_manifest.json"
        self.combined_chunks_path = self.chunks_dir / "chunks.parquet"
        self.embeddings_path = self.embeddings_dir / "embeddings.npy"

    def ensure(self) -> None:
        for path in (self.parsed_dir, self.chunks_dir, self.embeddings_dir, self.outputs_dir):
            path.mkdir(parents=True, exist_ok=True)

    def parsed_path(self, report: Path) -> Path:
        return self.parsed_dir / f"{report.stem}.parquet"

    def chunk_path(self, report: Path) -> Path:
        return self.chunks_dir / f"{report.stem}_chunks.parquet"

    def embedding_path(self, report_id: str) -> Path:
        return self.embeddings_dir / f"{report_id}_embeddings.npy"

    def read_rows(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        return read_table(path).to_dict(orient="records")

    def write_rows(self, path: Path, rows: list[dict]) -> int:
        frame = pd.DataFrame(rows)
        write_table(path, frame)
        return len(frame)

    def write_report_manifest(self, records: list[ReportCacheRecord]) -> None:
        self.report_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_manifest_path.write_text(
            json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_embedding_manifest(self) -> dict:
        if not self.embedding_manifest_path.exists():
            return {"chunk_ids": [], "report_embeddings": {}}
        return json.loads(self.embedding_manifest_path.read_text(encoding="utf-8"))

    def write_embedding_manifest(self, chunk_ids: list[str], report_embeddings: dict[str, str]) -> None:
        self.embedding_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_manifest_path.write_text(
            json.dumps(
                {"chunk_ids": chunk_ids, "embedding_count": len(chunk_ids), "report_embeddings": report_embeddings},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_embeddings(self) -> np.ndarray | None:
        if not self.embeddings_path.exists():
            return None
        return np.load(self.embeddings_path)

    def save_embeddings(self, embeddings: np.ndarray) -> None:
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, embeddings)
