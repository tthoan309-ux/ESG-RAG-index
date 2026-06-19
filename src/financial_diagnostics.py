from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from .financial_config import FinancialIndicatorConfig
from .storage import write_table


def write_financial_null_diagnosis(financial: pd.DataFrame, financial_pages: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if financial.empty:
        diagnosis = pd.DataFrame()
        write_table(output_path, diagnosis)
        return diagnosis

    indicators = FinancialIndicatorConfig().load()
    page_text = (
        financial_pages.groupby(["company", "year"], dropna=False)["text"]
        .apply(lambda values: "\n".join(map(str, values)))
        .reset_index()
    )
    page_text["normalized_text"] = page_text["text"].map(_normalize)
    text_map = {(str(row.company), int(row.year)): row.normalized_text for _, row in page_text.iterrows() if pd.notna(row.year)}

    rows: list[dict] = []
    for _, row in financial.iterrows():
        indicator = indicators.get(str(row["indicator_id"]))
        if indicator is None:
            reason = "unknown_indicator"
            alias_present = False
        else:
            aliases = [_normalize(alias) for alias in (indicator.aliases or (indicator.label_vi,))]
            key = (str(row["firm"]), int(row["year"])) if pd.notna(row["year"]) else ("", 0)
            text = text_map.get(key, "")
            alias_present = any(alias and alias in text for alias in aliases)
            if pd.notna(row.get("value")):
                reason = "extracted"
            elif not text:
                reason = "no_financial_pages_detected"
            elif alias_present:
                reason = "alias_present_but_number_not_extracted"
            else:
                reason = "alias_absent_from_financial_pages"
        rows.append(
            {
                "firm": row.get("firm"),
                "year": row.get("year"),
                "indicator_id": row.get("indicator_id"),
                "indicator_name": row.get("indicator_name"),
                "value_present": pd.notna(row.get("value")),
                "alias_present_on_financial_pages": alias_present,
                "null_reason": reason,
                "confidence": row.get("confidence", 0.0),
                "validation_flag": row.get("validation_flag", ""),
            }
        )
    diagnosis = pd.DataFrame(rows)
    write_table(output_path, diagnosis)
    return diagnosis


def _normalize(text: str) -> str:
    raw = str(text).lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFKD", raw)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", stripped).strip()
