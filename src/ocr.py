from __future__ import annotations

from pathlib import Path

import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
def ocr_pdf(path: Path, language: str = "vie+eng", dpi: int = 250) -> list[dict]:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "OCR requires optional dependencies. Install requirements-ocr.txt and make sure "
            "Tesseract OCR plus Poppler are available on PATH."
        ) from exc

    from .parser import infer_company_year, normalize_text

    company, year = infer_company_year(path)
    pages: list[dict] = []
    images = convert_from_path(str(path), dpi=dpi)
    for idx, image in enumerate(images, start=1):
        text = normalize_text(pytesseract.image_to_string(image, lang=language))
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
    return pages

