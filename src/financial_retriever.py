from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from .config import PipelinePaths, RetrievalConfig
from .financial_config import FinancialIndicator, FinancialIndicatorConfig
from .financial_diagnostics import write_financial_null_diagnosis
from .financial_extractor import FinancialValueExtractor
from .financial_validator import attach_validation_flags, validate_financial_dataset
from .manifest_manager import ManifestManager
from .progress import progress
from .retriever import VectorStore
from .storage import read_table, write_table


FINANCIAL_COLUMNS = [
    "firm",
    "year",
    "indicator_id",
    "indicator_name",
    "value",
    "currency",
    "unit",
    "source_report",
    "page",
    "confidence",
    "evidence",
    "extraction_method",
    "retrieval_score",
    "warehouse_group",
    "cache_hit",
    "warehouse_hit",
    "validation_flag",
]


@dataclass(frozen=True)
class FinancialResult:
    financial: pd.DataFrame
    runtime: pd.DataFrame
    quality: pd.DataFrame
    validation: pd.DataFrame


class FinancialRetriever:
    def __init__(
        self,
        store: VectorStore,
        manifest: ManifestManager,
        config: RetrievalConfig | None = None,
        indicator_config: FinancialIndicatorConfig | None = None,
        warehouse_dir: Path | None = None,
        cache_dir: Path | None = None,
        rebuild: bool = False,
    ):
        self.store = store
        self.manifest = manifest
        self.config = config or RetrievalConfig(top_k=8, warehouse_top_k=40)
        self.indicator_config = indicator_config or FinancialIndicatorConfig()
        self.warehouse_dir = warehouse_dir or PipelinePaths().financial_warehouse
        self.cache_dir = cache_dir or PipelinePaths().financial_cache
        self.rebuild = rebuild
        self.extractor = FinancialValueExtractor()
        self.config_hash = financial_config_hash(self.config)

    def export_financials(
        self,
        firm_years: list[tuple[str, int | None]],
        csv_path: Path | None = None,
        parquet_path: Path | None = None,
        runtime_path: Path | None = None,
        quality_path: Path | None = None,
    ) -> FinancialResult:
        paths = PipelinePaths()
        rows: list[dict] = []
        runtime_rows: list[dict] = []

        groups = self.indicator_config.by_group()
        for firm, year in progress(firm_years, total=len(firm_years), desc="Financials", unit="firm-year"):
            group_warehouses = {
                group_id: self._group_warehouse(firm, year, group_id, indicators)
                for group_id, indicators in groups.items()
            }
            for group_id, indicators in groups.items():
                chunks, warehouse_hit, retrieval_time = group_warehouses[group_id]
                for indicator in indicators:
                    start = perf_counter()
                    row, cache_hit = self._extract_indicator(firm, year, indicator, chunks, warehouse_hit)
                    rows.append(row)
                    runtime_rows.append(
                        {
                            "firm": firm,
                            "year": year,
                            "indicator": indicator.indicator_id,
                            "warehouse_group": group_id,
                            "retrieval_time": round(retrieval_time, 6),
                            "extraction_time": round(perf_counter() - start, 6),
                            "confidence": row.get("confidence", 0.0),
                            "cache_hit": cache_hit,
                            "warehouse_hit": warehouse_hit,
                        }
                    )

        financial = pd.DataFrame(rows, columns=FINANCIAL_COLUMNS)
        validation = validate_financial_dataset(financial)
        financial = attach_validation_flags(financial, validation)
        financial = self._normalize_output_schema(financial)
        quality = self._quality(financial, runtime_rows, validation)

        write_table(csv_path or paths.financial_dataset_csv, financial)
        write_table(parquet_path or paths.financial_dataset_parquet, financial)
        write_table(runtime_path or paths.financial_runtime, pd.DataFrame(runtime_rows))
        write_table(quality_path or paths.financial_quality, quality)
        write_table(paths.root / "outputs" / "pipeline_artifacts" / "financial_validation.csv", validation)
        if paths.financial_pages.exists():
            write_financial_null_diagnosis(
                financial,
                read_table(paths.financial_pages),
                paths.root / "outputs" / "pipeline_artifacts" / "financial_null_diagnosis.csv",
            )
        return FinancialResult(financial, pd.DataFrame(runtime_rows), quality, validation)

    def _normalize_output_schema(self, financial: pd.DataFrame) -> pd.DataFrame:
        output = financial.copy()
        for column in ("firm", "indicator_id", "indicator_name", "currency", "unit", "source_report", "page", "evidence", "extraction_method", "warehouse_group", "validation_flag"):
            output[column] = output[column].fillna("").astype(str)
        for column in ("value", "confidence", "retrieval_score"):
            output[column] = pd.to_numeric(output[column], errors="coerce")
        output["year"] = pd.to_numeric(output["year"], errors="coerce").astype("Int64")
        output["cache_hit"] = output["cache_hit"].fillna(False).astype(bool)
        output["warehouse_hit"] = output["warehouse_hit"].fillna(False).astype(bool)
        output["validation_flag"] = output.apply(self._row_anomaly_flag, axis=1)
        return output

    def _row_anomaly_flag(self, row: pd.Series) -> str:
        flags = [flag for flag in str(row.get("validation_flag", "")).split(";") if flag.strip()]
        value = row.get("value")
        if pd.isna(value):
            return "; ".join(flags)
        indicator_id = str(row.get("indicator_id", ""))
        currency = str(row.get("currency", ""))
        numeric = abs(float(value))
        if currency == "VND" and numeric > 1e18:
            flags.append("EXTREME_VND_VALUE")
        if indicator_id in {"OUTSTANDING_SHARES", "TREASURY_SHARES"} and numeric > 1e11:
            flags.append("EXTREME_SHARE_COUNT")
        if indicator_id == "EMPLOYEES" and numeric > 1e7:
            flags.append("EXTREME_EMPLOYEE_COUNT")
        return "; ".join(dict.fromkeys(flags))

    def _group_warehouse(
        self,
        firm: str,
        year: int | None,
        group_id: str,
        indicators: list[FinancialIndicator],
    ) -> tuple[list[dict], bool, float]:
        start = perf_counter()
        path = self._warehouse_path(firm, year, group_id)
        if path.exists() and not self.rebuild:
            return read_table(path).to_dict(orient="records"), True, 0.0

        candidate_idx = self.store._candidate_indexes(company=firm, year=year)
        if not candidate_idx:
            return [], False, perf_counter() - start

        query = self._group_query(group_id, indicators)
        dense = self.store._embedding_search(query, candidate_idx, top_k=max(self.config.prefetch_k, 30))
        keyword = self.store._bm25.search(query, candidate_idx, top_k=self.config.warehouse_top_k)
        chunks = self._merge_and_rank(dense, keyword)[: self.config.warehouse_top_k]
        for rank, chunk in enumerate(chunks, start=1):
            chunk["financial_rank"] = rank
            chunk["warehouse_group"] = group_id
            chunk["retrieval_score"] = float(chunk.get("financial_score", chunk.get("embedding_score", 0.0)) or 0.0)
        write_table(path, pd.DataFrame(chunks))
        return chunks, False, perf_counter() - start

    def _extract_indicator(
        self,
        firm: str,
        year: int | None,
        indicator: FinancialIndicator,
        chunks: list[dict],
        warehouse_hit: bool,
    ) -> tuple[dict, bool]:
        cache_path = self._indicator_cache_path(firm, year, indicator.indicator_id)
        if cache_path.exists() and not self.rebuild:
            cached = read_table(cache_path)
            if not cached.empty:
                row = cached.iloc[0].to_dict()
                row["cache_hit"] = True
                row["warehouse_hit"] = warehouse_hit
                return row, True

        best_row = self._empty_row(firm, year, indicator, warehouse_hit)
        best_confidence = -1.0
        for chunk in chunks[: self.config.top_k]:
            value = self.extractor.extract(str(chunk.get("text", "")), indicator, report_year=year)
            if value is None:
                continue
            combined_confidence = min(1.0, (value.confidence * 0.75) + (self._retrieval_confidence(chunk) * 0.25))
            if combined_confidence <= best_confidence:
                continue
            best_confidence = combined_confidence
            best_row = {
                "firm": firm,
                "year": value.year or year,
                "indicator_id": indicator.indicator_id,
                "indicator_name": indicator.label_vi,
                "value": value.value,
                "currency": value.currency,
                "unit": value.unit,
                "source_report": chunk.get("source_file", ""),
                "page": chunk.get("page", ""),
                "confidence": round(combined_confidence, 4),
                "evidence": value.evidence,
                "extraction_method": value.extraction_method,
                "retrieval_score": round(float(chunk.get("retrieval_score", 0.0) or 0.0), 6),
                "warehouse_group": indicator.statement_group,
                "cache_hit": False,
                "warehouse_hit": warehouse_hit,
                "validation_flag": "",
            }
        write_table(cache_path, pd.DataFrame([best_row]))
        return best_row, False

    def _empty_row(self, firm: str, year: int | None, indicator: FinancialIndicator, warehouse_hit: bool) -> dict:
        return {
            "firm": firm,
            "year": year,
            "indicator_id": indicator.indicator_id,
            "indicator_name": indicator.label_vi,
            "value": None,
            "currency": "VND" if indicator.normalization_rules.get("convert_to_vnd", False) else "",
            "unit": "",
            "source_report": "",
            "page": "",
            "confidence": 0.0,
            "evidence": "",
            "extraction_method": "none",
            "retrieval_score": 0.0,
            "warehouse_group": indicator.statement_group,
            "cache_hit": False,
            "warehouse_hit": warehouse_hit,
            "validation_flag": "",
        }

    def _group_query(self, group_id: str, indicators: list[FinancialIndicator]) -> str:
        group_headers = {
            "balance_sheet": "bảng cân đối kế toán balance sheet tài sản nợ phải trả vốn chủ sở hữu",
            "income_statement": "báo cáo kết quả hoạt động kinh doanh income statement doanh thu lợi nhuận chi phí",
            "shares": "cổ phiếu đang lưu hành cổ phiếu quỹ outstanding shares treasury shares",
            "employees": "lao động nhân viên chi phí đào tạo employees training expense",
        }
        return " ".join([group_headers.get(group_id, group_id), *(indicator.query_text for indicator in indicators)])

    def _merge_and_rank(self, dense: list[dict], keyword: list[dict]) -> list[dict]:
        merged: dict[str, dict] = {}
        for item in dense + keyword:
            key = str(item.get("chunk_id") or item.get("chunk_index"))
            existing = merged.setdefault(key, dict(item))
            existing.update({k: v for k, v in item.items() if k.endswith("_score") or k.endswith("_rank")})
        for item in merged.values():
            bm25_rank = float(item.get("bm25_rank", self.config.warehouse_top_k + 1))
            embedding_rank = float(item.get("embedding_rank", self.config.warehouse_top_k + 1))
            item["financial_score"] = (1.0 / (60.0 + bm25_rank)) + (1.0 / (60.0 + embedding_rank))
        return sorted(merged.values(), key=lambda item: item.get("financial_score", 0.0), reverse=True)

    def _retrieval_confidence(self, chunk: dict) -> float:
        score = float(chunk.get("financial_score", 0.0) or 0.0)
        return min(max(score * 35.0, 0.0), 1.0)

    def _warehouse_path(self, firm: str, year: int | None, group_id: str) -> Path:
        return self.warehouse_dir / f"{firm}_{year}_{group_id}_{self.config_hash}.parquet"

    def _indicator_cache_path(self, firm: str, year: int | None, indicator_id: str) -> Path:
        return self.cache_dir / f"{firm}_{year}_{indicator_id}_{self.config_hash}.parquet"

    def _quality(self, financial: pd.DataFrame, runtime_rows: list[dict], validation: pd.DataFrame) -> pd.DataFrame:
        runtime = pd.DataFrame(runtime_rows)
        summary = (
            financial.groupby("indicator_id", dropna=False)
            .agg(
                observations=("indicator_id", "size"),
                extracted=("value", lambda values: int(values.notna().sum())),
                avg_confidence=("confidence", "mean"),
                table_extractions=("extraction_method", lambda values: int((values == "table").sum())),
            )
            .reset_index()
        )
        summary["coverage"] = (summary["extracted"] / summary["observations"]).round(4)
        summary["avg_confidence"] = summary["avg_confidence"].round(4)
        if not runtime.empty:
            rt = runtime.groupby("indicator", dropna=False).agg(
                avg_retrieval_time=("retrieval_time", "mean"),
                avg_extraction_time=("extraction_time", "mean"),
                cache_hit_rate=("cache_hit", "mean"),
                warehouse_hit_rate=("warehouse_hit", "mean"),
            )
            summary = summary.merge(rt, left_on="indicator_id", right_index=True, how="left")
        failed_rules = validation.loc[validation["status"] == "FAIL"].groupby("rule").size().rename("failed_count").reset_index()
        if not failed_rules.empty:
            failed_rules["indicator_id"] = "VALIDATION"
            failed_rules["observations"] = len(financial)
            summary = pd.concat([summary, failed_rules], ignore_index=True, sort=False)
        return summary


def financial_config_hash(config: RetrievalConfig) -> str:
    payload: dict[str, Any] = {
        "top_k": config.top_k,
        "prefetch_k": config.prefetch_k,
        "warehouse_top_k": config.warehouse_top_k,
        "financial_engine": "v1",
        "retrieval_scope": "financial_pages_only_v1",
        "group_warehouse": True,
        "table_priority": True,
        "normalize_vnd": True,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
