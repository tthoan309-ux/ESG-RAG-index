from __future__ import annotations

import re
from pathlib import Path

from .config import ChunkingConfig
from .io_utils import read_jsonl, write_jsonl


TOKEN_RE = re.compile(r"\S+")


def token_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in TOKEN_RE.finditer(text)]


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    spans = token_spans(text)
    if not spans:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0
    while start < len(spans):
        end = min(start + chunk_size, len(spans))
        char_start = spans[start][0]
        char_end = spans[end - 1][1]
        chunks.append(text[char_start:char_end].strip())
        if end == len(spans):
            break
        start = end - overlap
    return chunks


def chunk_pages(parsed_files: list[Path], output_path: Path, config: ChunkingConfig) -> list[dict]:
    rows: list[dict] = []
    chunk_no = 1
    for parsed_file in parsed_files:
        for page in read_jsonl(parsed_file):
            for text in split_text(page["text"], config.chunk_size, config.overlap):
                rows.append(
                    {
                        "chunk_id": f"{page['company']}_{page['year']}_{chunk_no:06d}",
                        "company": page["company"],
                        "year": page["year"],
                        "source_file": page["source_file"],
                        "page": page["page"],
                        "section": page.get("section", ""),
                        "text": text,
                    }
                )
                chunk_no += 1
    write_jsonl(output_path, rows)
    return rows

