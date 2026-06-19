from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PipelinePaths
from .financial_pages import FinancialPageParser
from .manifest_manager import sha256_file
from .parser import infer_company_year, normalize_text
from .progress import progress
from .storage import read_table, write_table


@dataclass(frozen=True)
class FinancialOCRConfig:
    mode: str = "candidates"
    language: str = "vie+eng"
    dpi: int = 300
    neighbor_window: int = 1
    force: bool = False
    limit: int | None = None
    workers: int = 4
    include_glob: str | None = None


class FinancialOCRRunner:
    def __init__(self, paths: PipelinePaths | None = None):
        self.paths = paths or PipelinePaths()
        self.cache_dir = self.paths.root / "outputs" / "pipeline_artifacts" / "financial_ocr_cache"
        self.output_path = self.paths.root / "data" / "financial_ocr_pages.parquet"
        self.runtime_path = self.paths.root / "outputs" / "pipeline_artifacts" / "financial_ocr_runtime.csv"

    def run(self, config: FinancialOCRConfig) -> pd.DataFrame:
        reports = sorted(self.paths.raw_reports.glob("*.pdf"))
        if config.include_glob:
            reports = [report for report in reports if report.match(config.include_glob) or report.name.lower().find(config.include_glob.lower()) >= 0]
        if config.limit:
            reports = reports[: config.limit]
        candidate_pages = self._candidate_pages(config)
        rows: list[dict] = []
        runtime_rows: list[dict] = []

        jobs: list[tuple[Path, str, int]] = []
        for report in reports:
            company, year = infer_company_year(report)
            pages = candidate_pages.get(report.stem, set())
            if config.mode == "all":
                pages = set(range(1, self._page_count(report) + 1))
            if not pages:
                continue
            report_hash = sha256_file(report)
            for page_number in sorted(pages):
                jobs.append((report, report_hash, page_number))

        max_workers = max(int(config.workers or 1), 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(self._run_job, report, report_hash, page_number, config): (report, page_number) for report, report_hash, page_number in jobs}
            for future in progress(as_completed(future_map), total=len(future_map), desc="Financial OCR", unit="page"):
                row, runtime = future.result()
                runtime_rows.append(runtime)
                if row:
                    rows.append(row)

        frame = self._merge_existing(pd.DataFrame(rows))
        if not frame.empty:
            frame = frame.sort_values(["company", "year", "page_number"]).drop_duplicates(["source_file", "page_number"], keep="last")
        write_table(self.output_path, frame)
        write_table(self.runtime_path, self._merge_runtime(pd.DataFrame(runtime_rows)))
        return frame

    def _merge_existing(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.output_path.exists():
            return frame
        existing = read_table(self.output_path)
        if existing.empty:
            return frame
        if frame.empty:
            return existing
        return pd.concat([existing, frame], ignore_index=True)

    def _merge_runtime(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.runtime_path.exists():
            return frame
        existing = read_table(self.runtime_path)
        if existing.empty:
            return frame
        if frame.empty:
            return existing
        return pd.concat([existing, frame], ignore_index=True)

    def _run_job(self, report: Path, report_hash: str, page_number: int, config: FinancialOCRConfig) -> tuple[dict | None, dict]:
        company, year = infer_company_year(report)
        start = time.perf_counter()
        row, cache_hit, error = self._ocr_page(report, report_hash, page_number, config)
        return row, {
            "report": report.name,
            "company": company,
            "year": year,
            "page": page_number,
            "cache_hit": cache_hit,
            "runtime_seconds": round(time.perf_counter() - start, 6),
            "error": error,
        }

    def _candidate_pages(self, config: FinancialOCRConfig) -> dict[str, set[int]]:
        parser = FinancialPageParser(paths=self.paths, neighbor_window=config.neighbor_window)
        pages = parser.build(rebuild=False).pages
        by_report: dict[str, set[int]] = {}
        if pages.empty:
            return by_report
        for _, row in pages.iterrows():
            source_file = str(row.get("source_file", ""))
            stem = Path(source_file).stem
            try:
                page_number = int(row.get("page_number"))
            except Exception:
                continue
            by_report.setdefault(stem, set()).add(page_number)
            for nearby in range(max(1, page_number - config.neighbor_window), page_number + config.neighbor_window + 1):
                by_report[stem].add(nearby)
        return by_report

    def _ocr_page(self, report: Path, report_hash: str, page_number: int, config: FinancialOCRConfig) -> tuple[dict | None, bool, str]:
        cache_path = self.cache_dir / report_hash / f"page_{page_number:04d}.json"
        if cache_path.exists() and not config.force:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return payload if payload.get("text") else None, True, ""
        try:
            text = self._ocr_pdf_page(report, page_number, config)
            company, year = infer_company_year(report)
            payload = {
                "company": company,
                "year": year,
                "page_number": page_number,
                "source_file": report.name,
                "text": text,
                "ocr_source": "financial_targeted_ocr",
                "ocr_dpi": config.dpi,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return payload if text else None, False, ""
        except Exception as exc:
            return None, False, f"{type(exc).__name__}: {exc}"

    def _ocr_pdf_page(self, report: Path, page_number: int, config: FinancialOCRConfig) -> str:
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError as exc:
            raise RuntimeError("Financial OCR requires pdf2image and pytesseract. Install requirements-ocr.txt.") from exc
        tesseract_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if tesseract_path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)
        images = convert_from_path(str(report), dpi=config.dpi, first_page=page_number, last_page=page_number)
        if not images:
            return ""
        return normalize_text(pytesseract.image_to_string(images[0], lang=config.language))

    def _page_count(self, report: Path) -> int:
        try:
            import pdfplumber

            with pdfplumber.open(report) as pdf:
                return len(pdf.pages)
        except Exception:
            parsed = self.paths.parsed_reports / f"{report.stem}.parquet"
            if parsed.exists():
                frame = read_table(parsed)
                col = "page_number" if "page_number" in frame.columns else "page"
                return int(pd.to_numeric(frame[col], errors="coerce").max())
            return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted OCR for financial statement pages.")
    parser.add_argument("--mode", choices=["candidates", "all"], default="candidates")
    parser.add_argument("--language", default="vie+eng")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--neighbor-window", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include-glob", default=None, help="Optional report filename glob or substring, e.g. BSR_2024 or '*_2024.pdf'.")
    args = parser.parse_args()
    frame = FinancialOCRRunner().run(
        FinancialOCRConfig(
            mode=args.mode,
            language=args.language,
            dpi=args.dpi,
            neighbor_window=args.neighbor_window,
            force=args.force,
            limit=args.limit,
            workers=args.workers,
            include_glob=args.include_glob,
        )
    )
    print(f"Wrote {len(frame)} targeted financial OCR pages to {FinancialOCRRunner().output_path}")


if __name__ == "__main__":
    main()
