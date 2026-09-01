from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_CORPUS_COLUMNS = {
    "source_document_id", "source_segment_id", "ticker", "year", "text",
    "corpus_status", "provenance_status",
}


def read_corpus(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        frame = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported corpus format: {path.suffix}")
    missing = sorted(REQUIRED_CORPUS_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Corpus contract missing: {', '.join(missing)}")
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["text"] = frame["text"].fillna("").astype(str)
    if frame[["ticker", "year"]].isna().any().any() or frame["ticker"].eq("").any():
        raise ValueError("ticker/year cannot be empty in the ESG source corpus.")
    return frame.sort_values(["ticker", "year", "source_document_id", "source_segment_id"]).reset_index(drop=True)

