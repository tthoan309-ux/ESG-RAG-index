from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .chunker import split_text
from .config import ChunkingConfig, EmbeddingConfig, PipelinePaths
from .embedder import HashingEmbedder
from .retriever import VectorStore
from .storage import read_table, write_table


FINANCIAL_PAGE_COLUMNS = [
    "company",
    "year",
    "page_number",
    "source_file",
    "statement_type",
    "detection_score",
    "matched_patterns",
    "ocr_parse_suspect",
    "text",
]


STATEMENT_PATTERNS = {
    "balance_sheet": (
        "bang can doi ke toan",
        "bao cao tinh hinh tai chinh",
        "tai san ngan han",
        "tai san dai han",
        "no phai tra",
        "von chu so huu",
        "tong cong tai san",
        "balance sheet",
        "statement of financial position",
    ),
    "income_statement": (
        "bao cao ket qua hoat dong kinh doanh",
        "ket qua hoat dong kinh doanh",
        "doanh thu thuan",
        "gia von hang ban",
        "loi nhuan gop",
        "loi nhuan sau thue",
        "income statement",
        "statement of profit or loss",
    ),
    "cash_flow": (
        "bao cao luu chuyen tien te",
        "luu chuyen tien thuan",
        "luu chuyen tien tu hoat dong",
        "tien va tuong duong tien cuoi nam",
        "cash flow statement",
        "statement of cash flows",
    ),
    "notes": (
        "thuyet minh bao cao tai chinh",
        "cac khoan phai thu",
        "hang ton kho",
        "vay va no thue tai chinh",
        "notes to the financial statements",
    ),
    "shares_employees": (
        "co phieu dang luu hanh",
        "co phieu quy",
        "von dieu le",
        "nguoi lao dong",
        "so lao dong",
        "outstanding shares",
        "treasury shares",
        "charter capital",
        "employees",
    ),
}


@dataclass(frozen=True)
class FinancialCorpus:
    pages: pd.DataFrame
    chunks: list[dict]
    embeddings: np.ndarray
    store: VectorStore


class FinancialPageParser:
    def __init__(
        self,
        paths: PipelinePaths | None = None,
        chunk_config: ChunkingConfig | None = None,
        embedder: HashingEmbedder | None = None,
        neighbor_window: int = 1,
    ):
        self.paths = paths or PipelinePaths()
        self.chunk_config = chunk_config or ChunkingConfig(chunk_size=900, overlap=120)
        self.embedder = embedder or HashingEmbedder(EmbeddingConfig())
        self.neighbor_window = neighbor_window

    def build(self, rebuild: bool = False) -> FinancialCorpus:
        if (
            not rebuild
            and self.paths.financial_pages.exists()
            and self.paths.financial_chunks.exists()
            and self.paths.financial_embeddings.exists()
        ):
            pages = read_table(self.paths.financial_pages)
            chunks = read_table(self.paths.financial_chunks).to_dict(orient="records")
            embeddings = np.load(self.paths.financial_embeddings)
            return FinancialCorpus(pages, chunks, embeddings, VectorStore(embeddings, chunks, self.embedder))

        pages = self._detect_pages()
        chunks = self._chunk_pages(pages)
        embeddings = self.embedder.encode([str(chunk["text"]) for chunk in chunks]) if chunks else np.empty((0, EmbeddingConfig().dimensions), dtype=np.float32)
        write_table(self.paths.financial_pages, pages)
        write_table(self.paths.financial_chunks, pd.DataFrame(chunks))
        self.paths.financial_embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.paths.financial_embeddings, embeddings)
        self._write_quality(pages, chunks)
        return FinancialCorpus(pages, chunks, embeddings, VectorStore(embeddings, chunks, self.embedder))

    def _detect_pages(self) -> pd.DataFrame:
        parsed_files = sorted(self.paths.parsed_reports.glob("*.parquet"))
        selected_rows: list[dict] = []
        for path in parsed_files:
            frame = read_table(path)
            if frame.empty:
                continue
            frame = frame.copy()
            page_col = "page_number" if "page_number" in frame.columns else "page"
            detections = []
            for idx, row in frame.iterrows():
                statement_type, score, matched = self._classify(str(row.get("text", "")))
                if score > 0:
                    detections.append((idx, statement_type, score, matched))
            include: dict[int, tuple[str, int, list[str]]] = {}
            for idx, statement_type, score, matched in detections:
                for nearby in range(max(0, idx - self.neighbor_window), min(len(frame), idx + self.neighbor_window + 1)):
                    old = include.get(nearby)
                    if old is None or score > old[1]:
                        include[nearby] = (statement_type, score, matched)
            for idx, (statement_type, score, matched) in sorted(include.items()):
                row = frame.iloc[idx]
                text = str(row.get("text", ""))
                selected_rows.append(
                    {
                        "company": row.get("company", ""),
                        "year": row.get("year"),
                        "page_number": row.get(page_col, ""),
                        "source_file": row.get("source_file", path.name),
                        "statement_type": statement_type,
                        "detection_score": score,
                        "matched_patterns": "; ".join(matched),
                        "ocr_parse_suspect": self._suspect(text, score),
                        "text": text,
                    }
                )
        if not selected_rows:
            pages = pd.DataFrame(columns=FINANCIAL_PAGE_COLUMNS)
        else:
            pages = pd.DataFrame(selected_rows, columns=FINANCIAL_PAGE_COLUMNS)
            pages = pages.drop_duplicates(["source_file", "page_number", "statement_type"]).reset_index(drop=True)
        return self._merge_targeted_ocr_pages(pages)

    def _merge_targeted_ocr_pages(self, pages: pd.DataFrame) -> pd.DataFrame:
        ocr_path = self.paths.root / "data" / "financial_ocr_pages.parquet"
        if not ocr_path.exists():
            return pages
        ocr_pages = read_table(ocr_path)
        if ocr_pages.empty:
            return pages
        rows: list[dict] = []
        existing_keys: set[tuple[str, int, str]] = set()
        for _, row in pages.iterrows():
            page_number = int(row.get("page_number", 0) or 0)
            statement_type = str(row.get("statement_type", ""))
            source_file = str(row.get("source_file", ""))
            key = (source_file, page_number, statement_type)
            existing_keys.add(key)
            match = ocr_pages.loc[
                (ocr_pages["source_file"].astype(str) == source_file)
                & (pd.to_numeric(ocr_pages["page_number"], errors="coerce").fillna(-1).astype(int) == page_number)
            ]
            record = row.to_dict()
            if not match.empty:
                ocr_text = str(match.iloc[-1].get("text", ""))
                if len(ocr_text) >= max(len(str(record.get("text", ""))) * 0.7, 100):
                    record["text"] = ocr_text
                    record["ocr_parse_suspect"] = self._suspect(ocr_text, int(record.get("detection_score", 1)))
            rows.append(record)
        for _, row in ocr_pages.iterrows():
            statement_type, score, matched = self._classify(str(row.get("text", "")))
            if score <= 0:
                statement_type, score, matched = "financial_ocr", 1, ["targeted_ocr_page"]
            source_file = str(row.get("source_file", ""))
            page_number = int(row.get("page_number", 0) or 0)
            key = (source_file, page_number, statement_type)
            if key in existing_keys:
                continue
            rows.append(
                {
                    "company": row.get("company", ""),
                    "year": row.get("year"),
                    "page_number": page_number,
                    "source_file": source_file,
                    "statement_type": statement_type,
                    "detection_score": score,
                    "matched_patterns": "; ".join(matched),
                    "ocr_parse_suspect": self._suspect(str(row.get("text", "")), score),
                    "text": row.get("text", ""),
                }
            )
        return pd.DataFrame(rows, columns=FINANCIAL_PAGE_COLUMNS).drop_duplicates(
            ["source_file", "page_number", "statement_type"], keep="last"
        )

    def _classify(self, text: str) -> tuple[str, int, list[str]]:
        normalized = _normalize(text)
        best_type = ""
        best_matches: list[str] = []
        best_score = 0
        for statement_type, patterns in STATEMENT_PATTERNS.items():
            matches = [pattern for pattern in patterns if pattern in normalized]
            score = len(matches)
            if score > best_score:
                best_type = statement_type
                best_matches = matches
                best_score = score
        return best_type, best_score, best_matches

    def _suspect(self, text: str, score: int) -> bool:
        clean = str(text).strip()
        if len(clean) < 250 and score > 0:
            return True
        if clean.count("\n") < 2 and len(clean) < 600 and score > 0:
            return True
        replacement_markers = clean.count("?") + clean.count("�")
        return replacement_markers > max(len(clean) * 0.02, 5)

    def _chunk_pages(self, pages: pd.DataFrame) -> list[dict]:
        chunks: list[dict] = []
        counters: dict[str, int] = {}
        for _, page in pages.iterrows():
            report_key = f"{page['company']}_{page['year']}"
            counters.setdefault(report_key, 0)
            for text in split_text(str(page["text"]), self.chunk_config.chunk_size, self.chunk_config.overlap):
                counters[report_key] += 1
                chunks.append(
                    {
                        "chunk_id": f"{report_key}_FIN_{counters[report_key]:06d}",
                        "company": page["company"],
                        "year": page["year"],
                        "source_file": page["source_file"],
                        "page": page["page_number"],
                        "section": page["statement_type"],
                        "statement_type": page["statement_type"],
                        "financial_page": True,
                        "text": text,
                    }
                )
        return chunks

    def _write_quality(self, pages: pd.DataFrame, chunks: list[dict]) -> None:
        if pages.empty:
            quality = pd.DataFrame(columns=["company", "year", "financial_pages", "financial_chunks", "suspect_pages"])
        else:
            page_summary = pages.groupby(["company", "year"], dropna=False).agg(
                financial_pages=("page_number", "nunique"),
                suspect_pages=("ocr_parse_suspect", "sum"),
                statement_types=("statement_type", lambda values: "; ".join(sorted(set(map(str, values))))),
            )
            chunk_frame = pd.DataFrame(chunks)
            chunk_summary = chunk_frame.groupby(["company", "year"], dropna=False).size().rename("financial_chunks")
            quality = page_summary.join(chunk_summary, how="left").fillna({"financial_chunks": 0}).reset_index()
        write_table(self.paths.financial_page_quality, quality)


def _normalize(text: str) -> str:
    raw = str(text).lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFKD", raw)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", stripped).strip()
