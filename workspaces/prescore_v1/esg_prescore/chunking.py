from __future__ import annotations

import hashlib
import re

import pandas as pd


TOKEN_RE = re.compile(r"\S+")


def _chunk_id(segment_id: str, start: int, end: int, text: str) -> str:
    value = f"{segment_id}|{start}|{end}|{text}"
    return f"chk_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def chunk_corpus(corpus: pd.DataFrame, target_words: int = 220, overlap_words: int = 40) -> pd.DataFrame:
    if target_words < 50:
        raise ValueError("target_words must be at least 50.")
    if overlap_words < 0 or overlap_words >= target_words:
        raise ValueError("overlap_words must be >= 0 and < target_words.")
    rows: list[dict] = []
    step = target_words - overlap_words
    ready = corpus[corpus["corpus_status"].eq("READY")]
    for source in ready.to_dict(orient="records"):
        tokens = TOKEN_RE.findall(str(source["text"]))
        for start in range(0, len(tokens), step):
            end = min(start + target_words, len(tokens))
            if end <= start:
                continue
            text = " ".join(tokens[start:end])
            rows.append({
                "chunk_id": _chunk_id(str(source["source_segment_id"]), start, end, text),
                "source_document_id": source["source_document_id"],
                "source_segment_id": source["source_segment_id"],
                "ticker": source["ticker"],
                "year": source["year"],
                "page_number": source.get("page_number"),
                "provenance_status": source["provenance_status"],
                "word_start": start,
                "word_end": end,
                "word_count": end - start,
                "text": text,
            })
            if end == len(tokens):
                break
    return pd.DataFrame(rows, columns=[
        "chunk_id", "source_document_id", "source_segment_id", "ticker", "year",
        "page_number", "provenance_status", "word_start", "word_end", "word_count", "text",
    ])
