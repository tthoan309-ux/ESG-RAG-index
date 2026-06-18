from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

from .cache_manager import CacheManager, ReportCacheRecord
from .chunker import split_text
from .config import ChunkingConfig
from .error_manager import ErrorRecorder, PipelineError
from .manifest_manager import ManifestManager, ReportIdentity
from .ocr_manager import OCRTask, run_ocr_tasks
from .parser import SUPPORTED_SUFFIXES, parse_report
from .progress import progress
from .storage import read_table, write_table


@dataclass(frozen=True)
class CorpusConfig:
    include_glob: str = "*"
    limit: int | None = None
    force_ocr: bool = False
    ocr_language: str = "vie+eng"
    ocr_threshold_chars: int = 500
    ocr_threshold_words_per_page: float = 20.0
    ocr_workers: int | None = None
    ocr_timeout_seconds: int = 60
    rebuild_parsed: bool = False
    rebuild_chunks: bool = False
    incremental: bool = False
    resume: bool = False


@dataclass
class CorpusResult:
    reports_found: int
    reports_processed: int
    reports_skipped: int
    ocr_reports: int
    failed_reports: int
    parsed_files: list[Path]
    chunk_files: list[Path]
    chunks: list[dict]
    records: list[ReportCacheRecord]
    pages_processed: int
    parsing_seconds: float
    ocr_seconds: float
    chunking_seconds: float
    ocr_pages: int = 0
    ocr_failed_pages: int = 0
    ocr_pages_per_hour: float = 0.0
    ocr_failure_rate: float = 0.0


class CorpusManager:
    def __init__(self, input_dir: Path, cache: CacheManager, manifest: ManifestManager, errors: ErrorRecorder):
        self.input_dir = input_dir
        self.cache = cache
        self.manifest = manifest
        self.errors = errors
        self.ocr_stems: set[str] = set()

    def discover_reports(self, config: CorpusConfig) -> list[Path]:
        reports = sorted(p for p in self.input_dir.glob(config.include_glob) if p.suffix.lower() in SUPPORTED_SUFFIXES)
        return reports[: config.limit] if config.limit else reports

    def process(self, config: CorpusConfig, chunk_config: ChunkingConfig) -> CorpusResult:
        reports = self.discover_reports(config)
        identities = [self.manifest.identity(report) for report in reports]

        parse_start = perf_counter()
        parsed_files, processed, skipped, ocr_seconds, ocr_pages, ocr_failed_pages, ocr_pages_per_hour, ocr_failure_rate = self._parse_reports(
            identities, config
        )
        parsing_seconds = perf_counter() - parse_start - ocr_seconds

        chunk_start = perf_counter()
        chunk_files, chunks, records, pages = self._chunk_reports(identities, parsed_files, config, chunk_config)
        chunking_seconds = perf_counter() - chunk_start

        self.cache.write_rows(self.cache.combined_chunks_path, chunks)
        self.cache.write_report_manifest(records)
        self.manifest.save()

        return CorpusResult(
            reports_found=len(reports),
            reports_processed=processed,
            reports_skipped=skipped,
            ocr_reports=sum(1 for record in records if record.used_ocr),
            failed_reports=len(self.errors.errors),
            parsed_files=parsed_files,
            chunk_files=chunk_files,
            chunks=chunks,
            records=records,
            pages_processed=pages,
            parsing_seconds=parsing_seconds,
            ocr_seconds=ocr_seconds,
            ocr_pages=ocr_pages,
            ocr_failed_pages=ocr_failed_pages,
            ocr_pages_per_hour=ocr_pages_per_hour,
            ocr_failure_rate=ocr_failure_rate,
            chunking_seconds=chunking_seconds,
        )

    def _parse_reports(self, identities: list[ReportIdentity], config: CorpusConfig) -> tuple[list[Path], int, int, float, int, int, float, float]:
        parsed_files: list[Path] = []
        ocr_tasks: list[OCRTask] = []
        processed = 0
        skipped = 0

        for identity in progress(identities, desc="Parsing", unit="report"):
            report = identity.path
            parsed_path = self.cache.parsed_path(report)
            can_skip = (
                not config.force_ocr
                and not config.rebuild_parsed
                and parsed_path.exists()
                and self.manifest.is_complete(identity, "parsed")
            )
            if can_skip:
                parsed_files.append(parsed_path)
                skipped += 1
                continue
            if config.incremental and not self.manifest.changed_or_new(identity) and parsed_path.exists():
                parsed_files.append(parsed_path)
                skipped += 1
                continue

            try:
                rows = parse_report(report, use_ocr=False, ocr_language=config.ocr_language)
                if report.suffix.lower() == ".pdf" and self._needs_ocr(rows, config):
                    ocr_tasks.append(
                        OCRTask(
                            report_path=report,
                            output_path=parsed_path,
                            language=config.ocr_language,
                            cache_dir=self.cache.root / "data" / "ocr_cache",
                            manifest_path=self.cache.root / "outputs" / "ocr_manifest.parquet",
                            error_path=self.cache.root / "outputs" / "ocr_errors.csv",
                            timeout_seconds=config.ocr_timeout_seconds,
                            resume=config.resume,
                        )
                    )
                    continue
                self._write_parsed(parsed_path, rows)
                self.manifest.mark(identity, "parsed")
                parsed_files.append(parsed_path)
                processed += 1
            except Exception as exc:
                self.errors.record(report, "parse", exc)

        ocr_start = perf_counter()
        ocr_pages = 0
        ocr_failed_pages = 0
        for result in run_ocr_tasks(ocr_tasks, workers=config.ocr_workers):
            identity = next((item for item in identities if str(item.path) == result.report_path), None)
            if result.error:
                self.errors.errors.append(PipelineError(result.report_path, "ocr", "OCRError", result.error, result.traceback or ""))
                continue
            parsed_files.append(Path(result.output_path))
            self.ocr_stems.add(Path(result.report_path).stem)
            ocr_pages += result.total_pages or result.page_count
            ocr_failed_pages += result.failed_pages
            if identity:
                self.manifest.mark(identity, "parsed")
            processed += 1
        ocr_seconds = perf_counter() - ocr_start if ocr_tasks else 0.0
        ocr_pages_per_hour = (ocr_pages / ocr_seconds * 3600.0) if ocr_seconds > 0 else 0.0
        ocr_failure_rate = (ocr_failed_pages / ocr_pages) if ocr_pages else 0.0
        return parsed_files, processed, skipped, ocr_seconds, ocr_pages, ocr_failed_pages, ocr_pages_per_hour, ocr_failure_rate

    def _chunk_reports(
        self,
        identities: list[ReportIdentity],
        parsed_files: list[Path],
        config: CorpusConfig,
        chunk_config: ChunkingConfig,
    ) -> tuple[list[Path], list[dict], list[ReportCacheRecord], int]:
        parsed_by_stem = {path.stem: path for path in parsed_files}
        chunk_files: list[Path] = []
        all_chunks: list[dict] = []
        records: list[ReportCacheRecord] = []
        pages_processed = 0

        for identity in progress(identities, desc="Chunking", unit="report"):
            report = identity.path
            parsed_path = parsed_by_stem.get(report.stem)
            if parsed_path is None or not parsed_path.exists():
                continue
            chunk_path = self.cache.chunk_path(report)
            if not config.rebuild_chunks and chunk_path.exists() and self.manifest.is_complete(identity, "chunked"):
                chunks = self.cache.read_rows(chunk_path)
            else:
                try:
                    chunks = self._chunk_one(parsed_path, chunk_config)
                    self.cache.write_rows(chunk_path, chunks)
                    self.manifest.mark(identity, "chunked")
                except Exception as exc:
                    self.errors.record(report, "chunk", exc)
                    continue

            pages = self.cache.read_rows(parsed_path)
            pages_processed += len(pages)
            all_chunks.extend(chunks)
            chunk_files.append(chunk_path)
            records.append(
                ReportCacheRecord(
                    report_id=identity.report_id,
                    source_file=report.name,
                    file_hash=identity.file_hash,
                    parsed_path=str(parsed_path),
                    chunk_path=str(chunk_path),
                    embedding_path=str(self.cache.embedding_path(identity.report_id)),
                    used_ocr=report.stem in self.ocr_stems,
                    page_count=len(pages),
                    text_length=sum(len(str(page.get("text", ""))) for page in pages),
                    chunk_count=len(chunks),
                )
            )

        return chunk_files, all_chunks, records, pages_processed

    def _chunk_one(self, parsed_path: Path, config: ChunkingConfig) -> list[dict]:
        rows: list[dict] = []
        chunk_no = 1
        for page in read_table(parsed_path).to_dict(orient="records"):
            for text in split_text(str(page["text"]), config.chunk_size, config.overlap):
                rows.append(
                    {
                        "chunk_id": f"{page['company']}_{page['year']}_{chunk_no:06d}",
                        "company": page["company"],
                        "year": page["year"],
                        "source_file": page.get("source_file", parsed_path.name),
                        "page": page.get("page_number", page.get("page", "")),
                        "section": page.get("section", ""),
                        "text": text,
                    }
                )
                chunk_no += 1
        return rows

    def _write_parsed(self, parsed_path: Path, rows: list[dict]) -> None:
        normalized = []
        for row in rows:
            normalized.append(
                {
                    "company": row.get("company", ""),
                    "year": row.get("year"),
                    "page_number": row.get("page", row.get("page_number", "")),
                    "source_file": row.get("source_file", ""),
                    "text": row.get("text", ""),
                }
            )
        write_table(parsed_path, pd.DataFrame(normalized))

    def _needs_ocr(self, rows: list[dict], config: CorpusConfig) -> bool:
        if config.force_ocr:
            return True
        page_count = max(len(rows), 1)
        text = " ".join(str(row.get("text", "")) for row in rows)
        char_count = len(text)
        words_per_page = len(text.split()) / page_count
        return char_count < config.ocr_threshold_chars or words_per_page < config.ocr_threshold_words_per_page
