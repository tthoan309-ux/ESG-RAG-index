from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import __version__
from .batching import build_scoring_rows, write_chatgpt_batches
from .chunking import chunk_corpus
from .codebook import canonical_hash, load_codebook
from .contracts import read_corpus
from .retrieval import retrieve_candidates


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else root / path


def run_prescore(
    corpus_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    workspace_root: str | Path,
) -> dict[str, Any]:
    root = Path(workspace_root)
    corpus_path, output_dir, config_path = Path(corpus_path), Path(output_dir), Path(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    codebook_path = _resolve(root, config["codebook_path"])
    prompt_path = _resolve(root, config["prompt_template_path"])
    codebook = load_codebook(codebook_path)
    corpus = read_corpus(corpus_path)
    chunks = chunk_corpus(corpus, **config["chunking"])
    candidates = retrieve_candidates(chunks, codebook, **config["retrieval"])
    scoring_rows, review_queue = build_scoring_rows(corpus, candidates, codebook)

    forbidden = {"score", "confidence", "reasoning", "disclosure_status"}
    collision = forbidden & set(scoring_rows.columns)
    if collision:
        raise AssertionError(f"Pre-score table contains scoring outputs: {sorted(collision)}")

    chunks.to_parquet(output_dir / "chunks.parquet", index=False)
    candidates.to_parquet(output_dir / "evidence_candidates.parquet", index=False)
    scoring_rows.to_csv(output_dir / "scoring_rows.csv", index=False)
    review_queue.to_csv(output_dir / "review_queue.csv", index=False)
    prompt = prompt_path.read_text(encoding="utf-8")
    batch_manifest = write_chatgpt_batches(
        output_dir, scoring_rows, candidates, codebook, prompt,
        max_firm_years_per_batch=int(config["batching"]["max_firm_years_per_batch"]),
    )
    batch_manifest.to_csv(output_dir / "batch_manifest.csv", index=False)
    manifest = {
        "pipeline": "esg-prescore",
        "pipeline_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(corpus_path),
        "input_sha256": _file_hash(corpus_path),
        "config_sha256": canonical_hash(config),
        "codebook_version": codebook["codebook_version"],
        "codebook_sha256": canonical_hash(codebook),
        "retrieval_mode": config["retrieval"]["mode"],
        "corpus_rows": len(corpus),
        "firm_years": int(corpus[["ticker", "year"]].drop_duplicates().shape[0]),
        "chunks": len(chunks),
        "evidence_candidates": len(candidates),
        "rows_ready_for_chatgpt": int(scoring_rows["pre_score_status"].eq("READY_FOR_CHATGPT").sum()),
        "rows_in_review_queue": len(review_queue),
        "batches": len(batch_manifest),
        "scoring_executed": False,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
