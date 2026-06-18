from __future__ import annotations

import re
from pathlib import Path
from html.parser import HTMLParser

import pdfplumber
from docx import Document

from .io_utils import write_jsonl


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".docx", ".html", ".htm", ".md"}


class _TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def get_text(self) -> str:
        return "\n".join(self.parts)


def infer_company_year(path: Path) -> tuple[str, int | None]:
    match = re.match(r"(?P<company>.+?)_(?P<year>\d{4})$", path.stem)
    if not match:
        return path.stem, None
    return match.group("company"), int(match.group("year"))


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(path: Path, use_ocr: bool = False, ocr_language: str = "vie+eng") -> list[dict]:
    company, year = infer_company_year(path)
    pages: list[dict] = []
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if text:
                pages.append(
                    {
                        "company": company,
                        "year": year,
                        "source_file": path.name,
                        "page": idx,
                        "section": "",
                        "text": text,
                    }
                )
    if pages or not use_ocr:
        return pages

    from .ocr import ocr_pdf

    return ocr_pdf(path, language=ocr_language)


def parse_docx(path: Path) -> list[dict]:
    company, year = infer_company_year(path)
    doc = Document(path)
    text = normalize_text("\n".join(p.text for p in doc.paragraphs))
    return [
        {
            "company": company,
            "year": year,
            "source_file": path.name,
            "page": 1,
            "section": "",
            "text": text,
        }
    ] if text else []


def parse_text_like(path: Path) -> list[dict]:
    company, year = infer_company_year(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in {".html", ".htm"}:
        parser = _TextHTMLParser()
        parser.feed(raw)
        raw = parser.get_text()
    text = normalize_text(raw)
    return [
        {
            "company": company,
            "year": year,
            "source_file": path.name,
            "page": 1,
            "section": "",
            "text": text,
        }
    ] if text else []


def parse_report(path: Path, use_ocr: bool = False, ocr_language: str = "vie+eng") -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, use_ocr=use_ocr, ocr_language=ocr_language)
    if suffix == ".docx":
        return parse_docx(path)
    if suffix in {".txt", ".html", ".htm", ".md"}:
        return parse_text_like(path)
    raise ValueError(f"Unsupported report format: {path}")


def parse_reports(
    input_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    include_glob: str = "*",
    use_ocr: bool = False,
    ocr_language: str = "vie+eng",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = sorted(p for p in input_dir.glob(include_glob) if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if limit:
        reports = reports[:limit]

    outputs: list[Path] = []
    for report in reports:
        rows = parse_report(report, use_ocr=use_ocr, ocr_language=ocr_language)
        output_path = output_dir / f"{report.stem}.jsonl"
        write_jsonl(output_path, rows)
        outputs.append(output_path)
    return outputs
