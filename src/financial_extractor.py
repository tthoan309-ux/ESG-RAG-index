from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable

from .financial_config import FinancialIndicator


NUMBER_RE = re.compile(r"(?<!\w)-?\(?\d[\d.,]*\)?")
YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")


@dataclass(frozen=True)
class FinancialValue:
    value: float
    currency: str
    unit: str
    year: int | None
    confidence: float
    evidence: str
    extraction_method: str


class FinancialTableParser:
    def candidate_lines(self, text: str, indicator: FinancialIndicator) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        aliases = [_normalize(alias) for alias in (indicator.aliases or (indicator.label_vi,))]
        output: list[str] = []
        for idx, line in enumerate(lines):
            normalized = _normalize(line)
            if any(alias and alias in normalized for alias in aliases):
                if line not in output:
                    output.append(line)
        return output

    def statement_unit(self, text: str) -> str | None:
        normalized = _normalize(text[:1500])
        patterns = (
            ("nghìn đồng", ("don vi tinh nghin dong", "dvt nghin dong", "nghin dong")),
            ("triệu đồng", ("don vi tinh trieu dong", "dvt trieu dong", "trieu dong")),
            ("tỷ đồng", ("don vi tinh ty dong", "dvt ty dong", "ty dong")),
            ("VND", ("vnd", "dong viet nam")),
        )
        for unit, needles in patterns:
            if any(needle in normalized for needle in needles):
                return unit
        return None


class FinancialValueExtractor:
    def __init__(self):
        self.table_parser = FinancialTableParser()

    def extract(self, text: str, indicator: FinancialIndicator, report_year: int | None = None) -> FinancialValue | None:
        table_unit = self.table_parser.statement_unit(text)
        candidates: list[tuple[str, str]] = [("table", line) for line in self.table_parser.candidate_lines(text, indicator)]
        if not candidates:
            candidates = [("text", segment) for segment in self._candidate_segments(text, indicator)]

        best: FinancialValue | None = None
        for method, segment in candidates:
            extracted = self._extract_from_segment(segment, indicator, report_year, table_unit, method)
            if extracted and (best is None or extracted.confidence > best.confidence):
                best = extracted
        return best

    def _candidate_segments(self, text: str, indicator: FinancialIndicator) -> list[str]:
        aliases = [_normalize(alias) for alias in (indicator.aliases or (indicator.label_vi,))]
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        output: list[str] = []
        for sentence in sentences:
            normalized = _normalize(sentence)
            if any(alias and alias in normalized for alias in aliases):
                output.append(sentence.strip())
        return output[:8]

    def _extract_from_segment(
        self,
        segment: str,
        indicator: FinancialIndicator,
        report_year: int | None,
        table_unit: str | None,
        method: str,
    ) -> FinancialValue | None:
        value_text = self._value_text(segment, indicator)
        numbers = NUMBER_RE.findall(value_text)
        if not numbers:
            return None
        statement_unit = table_unit if indicator.normalization_rules.get("convert_to_vnd", False) else None
        unit = statement_unit or self._detect_unit(segment, indicator.unit_candidates, indicator) or self._fallback_unit(indicator)
        currency = self._detect_currency(segment, unit, indicator)
        year = self._detect_year(segment) or report_year

        values: list[tuple[float, str]] = []
        for raw_number in numbers:
            parsed = parse_number(raw_number, unit)
            if parsed is None:
                continue
            if self._looks_like_table_code(parsed, indicator):
                continue
            normalized = normalize_value(parsed, unit, indicator.normalization_rules)
            values.append((normalized, raw_number))
        if not values:
            return None

        value, raw = self._choose_value(values, indicator)
        confidence = self._confidence(segment, indicator, unit, method)
        return FinancialValue(
            value=float(value),
            currency=currency,
            unit=unit,
            year=year,
            confidence=confidence,
            evidence=segment.strip()[:2500],
            extraction_method=method,
        )

    def _value_text(self, segment: str, indicator: FinancialIndicator) -> str:
        lower = segment.lower()
        positions = []
        for alias in indicator.aliases or (indicator.label_vi,):
            alias_lower = str(alias).lower()
            pos = lower.find(alias_lower)
            if pos >= 0:
                positions.append(pos + len(alias_lower))
        if positions:
            return segment[min(positions) :]
        normalized = _normalize(segment)
        for alias in indicator.aliases or (indicator.label_vi,):
            alias_norm = _normalize(alias)
            pos = normalized.find(alias_norm)
            if pos >= 0:
                return normalized[pos + len(alias_norm) :]
        return segment

    def _choose_value(self, values: list[tuple[float, str]], indicator: FinancialIndicator) -> tuple[float, str]:
        if indicator.indicator_id in {"TREASURY_SHARES"}:
            return min(values, key=lambda item: abs(item[0]))
        if indicator.indicator_id in {"EMPLOYEES", "OUTSTANDING_SHARES"}:
            return max(values, key=lambda item: item[0])
        return max(values, key=lambda item: abs(item[0]))

    def _detect_unit(self, segment: str, candidates: Iterable[str], indicator: FinancialIndicator) -> str | None:
        normalized = _normalize(segment)
        for unit in sorted(candidates, key=len, reverse=True):
            if _normalize(unit) in normalized:
                return unit
        if indicator.normalization_rules.get("convert_to_vnd", False):
            financial_patterns = (
                ("nghìn đồng", ("nghin dong", "thousand vnd")),
                ("triệu đồng", ("trieu dong", "million vnd")),
                ("tỷ đồng", ("ty dong", "billion vnd")),
                ("VND", (" vnd ", " dong ")),
            )
            for unit, needles in financial_patterns:
                if any(needle in f" {normalized} " for needle in needles):
                    return unit
            return None
        unit_patterns = (
            ("cổ phiếu", ("co phieu", "shares")),
            ("người", ("nguoi", "employees", "persons", "lao dong")),
        )
        for unit, needles in unit_patterns:
            if any(needle in f" {normalized} " for needle in needles):
                return unit
        return None

    def _looks_like_table_code(self, value: Decimal, indicator: FinancialIndicator) -> bool:
        if indicator.indicator_id in {"EMPLOYEES", "OUTSTANDING_SHARES", "TREASURY_SHARES"}:
            return False
        absolute = abs(value)
        return Decimal("1900") <= absolute <= Decimal("2099") or absolute < Decimal("1000")

    def _fallback_unit(self, indicator: FinancialIndicator) -> str:
        if indicator.indicator_id in {"EMPLOYEES"}:
            return "người"
        if indicator.indicator_id in {"OUTSTANDING_SHARES", "TREASURY_SHARES"}:
            return "cổ phiếu"
        return "VND"

    def _detect_currency(self, segment: str, unit: str, indicator: FinancialIndicator) -> str:
        if indicator.normalization_rules.get("convert_to_vnd", False):
            return "VND"
        if unit in {"cổ phiếu", "shares"}:
            return "SHARES"
        if unit in {"người", "employees", "persons"}:
            return "PERSONS"
        normalized = _normalize(segment)
        if "usd" in normalized:
            return "USD"
        return "VND"

    def _detect_year(self, segment: str) -> int | None:
        years = [int(match) for match in YEAR_RE.findall(segment)]
        return max(years) if years else None

    def _confidence(self, segment: str, indicator: FinancialIndicator, unit: str, method: str) -> float:
        normalized = _normalize(segment)
        alias_hit = any(_normalize(alias) in normalized for alias in indicator.aliases)
        unit_hit = _normalize(unit) in normalized
        score = 0.35
        if alias_hit:
            score += 0.25
        if unit_hit:
            score += 0.15
        if method == "table":
            score += 0.20
        if YEAR_RE.search(segment):
            score += 0.05
        return min(round(score, 4), 1.0)


def parse_number(raw: str, unit: str | None = None) -> Decimal | None:
    text = raw.strip()
    negative = text.startswith("-") or (text.startswith("(") and text.endswith(")"))
    text = text.strip("()").replace(" ", "")
    text = re.sub(r"[^\d.,-]", "", text)
    if not text or text in {"-", ".", ","}:
        return None

    decimal_unit = unit and _normalize(unit) in {"ty dong", "billion vnd"}
    if "," in text and "." in text:
        last_comma = text.rfind(",")
        last_dot = text.rfind(".")
        decimal_sep = "," if last_comma > last_dot else "."
        thousands_sep = "." if decimal_sep == "," else ","
        cleaned = text.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif decimal_unit and re.match(r"^-?\d+[,.]\d{1,3}$", text):
        cleaned = text.replace(",", ".")
    else:
        cleaned = text.replace(".", "").replace(",", "")

    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -value if negative else value


def normalize_value(value: Decimal, unit: str, rules: dict) -> Decimal:
    if not rules.get("convert_to_vnd", False):
        return value
    normalized_unit = _normalize(unit)
    multipliers = {
        "vnd": Decimal("1"),
        "dong": Decimal("1"),
        "đong": Decimal("1"),
        "nghin dong": Decimal("1000"),
        "thousand vnd": Decimal("1000"),
        "trieu dong": Decimal("1000000"),
        "million vnd": Decimal("1000000"),
        "ty dong": Decimal("1000000000"),
        "billion vnd": Decimal("1000000000"),
    }
    multiplier = multipliers.get(normalized_unit, Decimal("1"))
    return value * multiplier


def _normalize(text: str) -> str:
    raw = str(text).lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFKD", raw)
    text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()
