from __future__ import annotations

import os
import queue
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import pandas as pd

from .manifest_manager import sha256_file
from .parser import infer_company_year, normalize_text
from .progress import progress
from .storage import read_table, write_table


OCR_MANIFEST_COLUMNS = ["report_hash", "report_name", "total_pages", "completed_pages", "status", "last_updated"]
OCR_ERROR_COLUMNS = ["report", "page", "error", "traceback"]


@dataclass(frozen=True)
class OCRTask:
    report_path: Path
    output_path: Path
    language: str
    cache_dir: Path
    manifest_path: Path
    error_path: Path
    timeout_seconds: int = 60
    dpi: int = 250
    resume: bool = True


@dataclass(frozen=True)
class OCRPageJob:
    report_path: Path
    report_hash: str
    report_name: str
    page_number: int
    output_path: Path
    language: str
    timeout_seconds: int
    dpi: int


@dataclass(frozen=True)
class OCRResult:
    report_path: str
    output_path: str
    page_count: int
    text_length: int
    total_pages: int = 0
    completed_pages: int = 0
    failed_pages: int = 0
    runtime_seconds: float = 0.0
    pages_per_hour: float = 0.0
    failure_rate: float = 0.0
    error: str | None = None
    traceback: str | None = None


def default_ocr_workers() -> int:
    return max((os.cpu_count() or 2) - 1, 1)


def run_ocr_tasks(tasks: list[OCRTask], workers: int | None = None) -> list[OCRResult]:
    if not tasks:
        return []

    max_workers = max(workers or default_ocr_workers(), 1)
    return [runner.run(task) for runner in (ResumableOCRRunner(max_workers=max_workers),) for task in tasks]


class OCRManifest:
    def __init__(self, path: Path):
        self.path = path
        self.frame = self._load()

    def _load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=OCR_MANIFEST_COLUMNS)
        frame = read_table(self.path)
        for column in OCR_MANIFEST_COLUMNS:
            if column not in frame.columns:
                frame[column] = 0 if column in {"total_pages", "completed_pages"} else ""
        return frame[OCR_MANIFEST_COLUMNS]

    def update(self, report_hash: str, report_name: str, total_pages: int, completed_pages: int, status: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        matches = self.frame.index[self.frame["report_hash"] == report_hash].tolist()
        row = {
            "report_hash": report_hash,
            "report_name": report_name,
            "total_pages": int(total_pages),
            "completed_pages": int(completed_pages),
            "status": status,
            "last_updated": now,
        }
        if matches:
            for key, value in row.items():
                self.frame.loc[matches[0], key] = value
        else:
            self.frame = pd.concat([self.frame, pd.DataFrame([row])], ignore_index=True)
        self.save()

    def save(self) -> None:
        write_table(self.path, self.frame)


class OCRErrorLog:
    def __init__(self, path: Path):
        self.path = path

    def append(self, report: str, page: int, error: str, tb: str = "") -> None:
        row = pd.DataFrame([{"report": report, "page": page, "error": error, "traceback": tb}], columns=OCR_ERROR_COLUMNS)
        if self.path.exists():
            existing = pd.read_csv(self.path)
            row = pd.concat([existing, row], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(self.path, index=False)


class ResumableOCRRunner:
    def __init__(self, max_workers: int):
        self.max_workers = max_workers

    def run(self, task: OCRTask) -> OCRResult:
        started = time.perf_counter()
        manifest = OCRManifest(task.manifest_path)
        errors = OCRErrorLog(task.error_path)
        report_hash = sha256_file(task.report_path)
        report_cache = task.cache_dir / report_hash
        report_cache.mkdir(parents=True, exist_ok=True)

        try:
            total_pages = self._count_pages(task.report_path)
        except Exception as exc:
            tb = traceback.format_exc()
            errors.append(str(task.report_path), 0, f"{type(exc).__name__}: {exc}", tb)
            manifest.update(report_hash, task.report_path.name, 0, 0, "failed")
            return OCRResult(
                report_path=str(task.report_path),
                output_path=str(task.output_path),
                page_count=0,
                text_length=0,
                error=f"{type(exc).__name__}: {exc}",
                traceback=tb,
            )

        completed = self._completed_pages(report_cache, total_pages)
        if completed >= total_pages:
            self._assemble_report(task, report_cache, total_pages)
            manifest.update(report_hash, task.report_path.name, total_pages, total_pages, "completed")
            return self._result(task, total_pages, started, failed_pages=0)

        manifest.update(report_hash, task.report_path.name, total_pages, completed, "running")
        pending = [
            OCRPageJob(
                report_path=task.report_path,
                report_hash=report_hash,
                report_name=task.report_path.name,
                page_number=page_number,
                output_path=self._page_path(report_cache, page_number),
                language=task.language,
                timeout_seconds=task.timeout_seconds,
                dpi=task.dpi,
            )
            for page_number in range(1, total_pages + 1)
            if not self._page_path(report_cache, page_number).exists()
        ]

        failed_pages = 0
        for event in progress(
            self._run_page_jobs(pending, task.timeout_seconds, errors),
            total=len(pending),
            desc=f"OCR {task.report_path.name}",
            unit="page",
        ):
            if not event.get("ok", False):
                failed_pages += 1
            completed = self._completed_pages(report_cache, total_pages)
            status = "completed" if completed >= total_pages else "running"
            manifest.update(report_hash, task.report_path.name, total_pages, completed, status)

        completed = self._completed_pages(report_cache, total_pages)
        status = "completed" if completed >= total_pages else "partial"
        manifest.update(report_hash, task.report_path.name, total_pages, completed, status)
        self._assemble_report(task, report_cache, total_pages)
        return self._result(task, total_pages, started, failed_pages)

    def _run_page_jobs(self, jobs: list[OCRPageJob], timeout_seconds: int, errors: OCRErrorLog):
        ctx = get_context("spawn")
        waiting = list(jobs)
        active: dict[int, dict[str, Any]] = {}

        while waiting or active:
            while waiting and len(active) < self.max_workers:
                job = waiting.pop(0)
                result_queue = ctx.Queue(maxsize=1)
                process = ctx.Process(target=_ocr_page_worker, args=(job, result_queue))
                process.start()
                active[process.pid or id(process)] = {
                    "process": process,
                    "queue": result_queue,
                    "job": job,
                    "started": time.monotonic(),
                }

            for key, state in list(active.items()):
                process = state["process"]
                job = state["job"]
                elapsed = time.monotonic() - state["started"]
                if process.is_alive() and elapsed > timeout_seconds:
                    process.terminate()
                    process.join(timeout=5)
                    _write_failed_page(job, f"TimeoutError: OCR exceeded {timeout_seconds}s/page")
                    errors.append(str(job.report_path), job.page_number, f"TimeoutError: OCR exceeded {timeout_seconds}s/page", "")
                    active.pop(key, None)
                    yield {"ok": False, "page": job.page_number}
                    continue

                if process.exitcode is None:
                    continue

                process.join()
                result = _queue_get_nowait(state["queue"]) or {}
                if process.exitcode != 0 and not result:
                    _write_failed_page(job, f"WorkerExitError: OCR worker exited with code {process.exitcode}")
                    errors.append(
                        str(job.report_path),
                        job.page_number,
                        f"WorkerExitError: OCR worker exited with code {process.exitcode}",
                        "",
                    )
                    result = {"ok": False}
                elif not result.get("ok", False) and result.get("error"):
                    errors.append(str(job.report_path), job.page_number, str(result.get("error", "")), str(result.get("traceback", "")))
                active.pop(key, None)
                yield {"ok": bool(result.get("ok", False)), "page": job.page_number}

            time.sleep(0.1)

    def _count_pages(self, report_path: Path) -> int:
        try:
            from pdf2image import pdfinfo_from_path

            info = pdfinfo_from_path(str(report_path))
            return int(info["Pages"])
        except Exception:
            import pdfplumber

            with pdfplumber.open(report_path) as pdf:
                return len(pdf.pages)

    def _assemble_report(self, task: OCRTask, report_cache: Path, total_pages: int) -> None:
        rows = []
        for page_number in range(1, total_pages + 1):
            path = self._page_path(report_cache, page_number)
            if not path.exists():
                continue
            rows.extend(read_table(path).to_dict(orient="records"))
        rows.sort(key=lambda row: int(row.get("page_number", row.get("page", 0)) or 0))
        write_table(task.output_path, pd.DataFrame(rows))

    def _completed_pages(self, report_cache: Path, total_pages: int) -> int:
        return sum(1 for page_number in range(1, total_pages + 1) if self._page_path(report_cache, page_number).exists())

    def _page_path(self, report_cache: Path, page_number: int) -> Path:
        return report_cache / f"page_{page_number:05d}.parquet"

    def _result(self, task: OCRTask, total_pages: int, started: float, failed_pages: int) -> OCRResult:
        rows = read_table(task.output_path).to_dict(orient="records") if task.output_path.exists() else []
        runtime = time.perf_counter() - started
        pages_per_hour = (total_pages / runtime * 3600.0) if runtime > 0 else 0.0
        failed = sum(1 for row in rows if str(row.get("ocr_status", "")) == "failed")
        failure_rate = (failed / total_pages) if total_pages else 0.0
        return OCRResult(
            report_path=str(task.report_path),
            output_path=str(task.output_path),
            page_count=len(rows),
            text_length=sum(len(str(row.get("text", ""))) for row in rows),
            total_pages=total_pages,
            completed_pages=min(len(rows), total_pages),
            failed_pages=failed,
            runtime_seconds=runtime,
            pages_per_hour=pages_per_hour,
            failure_rate=failure_rate,
        )


def _queue_get_nowait(result_queue) -> dict | None:
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return None


def _ocr_page_worker(job: OCRPageJob, result_queue) -> None:
    try:
        from pdf2image import convert_from_path
        import pytesseract

        tesseract_cmd = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if tesseract_cmd.exists():
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)

        company, year = infer_company_year(job.report_path)
        images = convert_from_path(
            str(job.report_path),
            dpi=job.dpi,
            first_page=job.page_number,
            last_page=job.page_number,
        )
        text = ""
        if images:
            text = normalize_text(
                pytesseract.image_to_string(images[0], lang=job.language, timeout=job.timeout_seconds)
            )
        row = _page_row(job, company, year, text, "completed")
        write_table(job.output_path, pd.DataFrame([row]))
        result_queue.put({"ok": True, "page": job.page_number})
    except Exception as exc:
        tb = traceback.format_exc()
        _write_failed_page(job, f"{type(exc).__name__}: {exc}", tb)
        result_queue.put({"ok": False, "page": job.page_number, "error": f"{type(exc).__name__}: {exc}", "traceback": tb})


def _write_failed_page(job: OCRPageJob, error: str, tb: str = "") -> None:
    company, year = infer_company_year(job.report_path)
    row = _page_row(job, company, year, "", "failed", error)
    write_table(job.output_path, pd.DataFrame([row]))


def _page_row(
    job: OCRPageJob,
    company: str,
    year: int | None,
    text: str,
    status: str,
    error: str = "",
) -> dict:
    return {
        "company": company,
        "year": year,
        "page_number": job.page_number,
        "source_file": job.report_name,
        "text": text,
        "ocr_status": status,
        "ocr_error": error,
    }
