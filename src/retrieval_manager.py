from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import pandas as pd

from .config import PipelinePaths, RetrievalConfig
from .evidence_builder import build_evidence_bundle
from .export_evidence import EVIDENCE_COLUMNS
from .manifest_manager import ManifestManager, ReportIdentity
from .progress import progress
from .reranker import BGEReranker
from .retriever import VectorStore, tokenize
from .storage import read_table, write_table
from .topic_manager import TopicManager


@dataclass
class RetrievalResult:
    evidence: pd.DataFrame
    retrieval_count: int
    reused_count: int


@dataclass
class TopicWarehouse:
    topic_id: str
    topic_name: str
    chunks: list[dict]
    cache_hit: bool
    retrieval_runtime: float
    retrieval_calls: int


class RetrievalManager:
    def __init__(
        self,
        store: VectorStore,
        reranker: BGEReranker,
        config: RetrievalConfig,
        manifest: ManifestManager,
        cache_dir: Path | None = None,
        warehouse_dir: Path | None = None,
        runtime_path: Path | None = None,
        benchmark_path: Path | None = None,
        rebuild: bool = False,
        workers: int = 1,
    ):
        self.store = store
        self.reranker = reranker
        self.config = config
        self.manifest = manifest
        self.cache_dir = cache_dir or PipelinePaths().evidence_cache
        self.warehouse_dir = warehouse_dir or PipelinePaths().evidence_warehouse
        self.runtime_path = runtime_path or PipelinePaths().retrieval_runtime
        self.benchmark_path = benchmark_path or PipelinePaths().warehouse_benchmark
        self.rebuild = rebuild
        self.workers = workers
        self.config_hash = retrieval_config_hash(config)
        self.topic_manager = TopicManager()
        self._manifest_lock = Lock()

    def export_evidence(
        self,
        codebook: pd.DataFrame,
        firm_years: list[tuple[str, int | None]],
        output_path: Path | None = None,
    ) -> RetrievalResult:
        codebook = self.topic_manager.assign(codebook)
        rows: list[dict] = []
        runtime_rows: list[dict] = []
        benchmark_rows: list[dict] = []
        reused = 0

        if self.workers <= 1:
            iterator = (self._retrieve_firm_year(company, year, codebook) for company, year in firm_years)
            for group_rows, group_runtime, group_benchmark, group_reused in progress(
                iterator, total=len(firm_years), desc="Retrieving", unit="firm-year"
            ):
                rows.extend(group_rows)
                runtime_rows.extend(group_runtime)
                benchmark_rows.append(group_benchmark)
                reused += group_reused
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = [executor.submit(self._retrieve_firm_year, company, year, codebook) for company, year in firm_years]
                for future in progress(as_completed(futures), total=len(futures), desc="Retrieving", unit="firm-year"):
                    group_rows, group_runtime, group_benchmark, group_reused = future.result()
                    rows.extend(group_rows)
                    runtime_rows.extend(group_runtime)
                    benchmark_rows.append(group_benchmark)
                    reused += group_reused

        evidence = pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)
        write_table(output_path or PipelinePaths().evidence_dataset, evidence)
        write_table(self.runtime_path, pd.DataFrame(runtime_rows))
        write_table(self.benchmark_path, pd.DataFrame(benchmark_rows))
        self.manifest.mark_exported_for_all_current()
        self.manifest.save()
        return RetrievalResult(evidence, len(rows), reused)

    def _retrieve_firm_year(
        self, company: str, year: int | None, codebook: pd.DataFrame
    ) -> tuple[list[dict], list[dict], dict, int]:
        report_id = f"{company}_{year}"
        report_hash = self._report_hash(report_id)
        rows: list[dict] = []
        runtimes: list[dict] = []
        reused = 0
        retrieval_calls_after = 0
        runtime_before = 0.0
        runtime_after = 0.0
        warehouse_hits = 0
        warehouse_misses = 0

        topic_groups = {topic_id: group.copy() for topic_id, group in codebook.groupby("topic_id")}
        topic_warehouses: dict[str, TopicWarehouse] = {}

        for topic_id, group in topic_groups.items():
            topic = self._topic_warehouse(company, year, report_hash, topic_id, group)
            topic_warehouses[topic_id] = topic
            retrieval_calls_after += topic.retrieval_calls
            runtime_after += topic.retrieval_runtime
            if topic.cache_hit:
                warehouse_hits += len(group)
            else:
                warehouse_misses += len(group)

        for topic_id, group in topic_groups.items():
            topic = topic_warehouses[topic_id]
            topic_reuse_count = max(len(group) - 1, 0)
            for _, indicator in group.iterrows():
                start = perf_counter()
                cache_path = self._indicator_cache_path(report_hash, str(indicator.indicator_id))
                row = self._row_from_topic(company, year, indicator, topic, topic_reuse_count, warehouse_hits, warehouse_misses)
                self._write_indicator_cache(cache_path, row, report_hash, topic_id)
                rows.append(row)
                reused += int(topic.cache_hit)
                runtimes.append(
                    {
                        "report": report_id,
                        "indicator": indicator.indicator_id,
                        "topic_id": topic_id,
                        "runtime_seconds": round(perf_counter() - start, 6),
                        "retrieval_score": row.get("retrieval_score", 0.0),
                        "rerank_score": row.get("rerank_score", 0.0),
                        "reranker_used": False,
                        "topic_reuse_count": topic_reuse_count,
                        "warehouse_hits": warehouse_hits,
                        "warehouse_misses": warehouse_misses,
                    }
                )

        before = len(codebook)
        warehouse_hit_rate = warehouse_hits / max(warehouse_hits + warehouse_misses, 1) * 100
        benchmark = {
            "report": report_id,
            "retrieval_calls_before": before,
            "retrieval_calls_after": retrieval_calls_after,
            "warehouse_hit_rate": round(warehouse_hit_rate, 3),
            "runtime_before": round(runtime_before, 6),
            "runtime_after": round(runtime_after, 6),
            "topic_count": len(topic_groups),
            "indicator_count": len(codebook),
            "warehouse_hits": warehouse_hits,
            "warehouse_misses": warehouse_misses,
            "call_reduction_percent": round((1 - retrieval_calls_after / max(before, 1)) * 100, 3),
        }
        self._mark_retrieved(report_id)
        return rows, runtimes, benchmark, reused

    def _topic_warehouse(self, company: str, year: int | None, report_hash: str, topic_id: str, group: pd.DataFrame) -> TopicWarehouse:
        topic_name = str(group.iloc[0].get("topic_name", topic_id))
        path = self._topic_cache_path(report_hash, topic_id)
        if path.exists() and not self.rebuild:
            return TopicWarehouse(topic_id, topic_name, read_table(path).to_dict(orient="records"), True, 0.0, 0)

        legacy = self._legacy_topic_from_indicator_cache(company, year, group)
        if legacy:
            write_table(path, pd.DataFrame(legacy))
            return TopicWarehouse(topic_id, topic_name, legacy, True, 0.0, 0)

        start = perf_counter()
        query = self._topic_query(group)
        if self.config.retrieval_mode == "bm25":
            candidate_idx = self.store._candidate_indexes(company=company, year=year)
            chunks = self.store._bm25.search(query, candidate_idx, self.config.warehouse_top_k)
            for item in chunks:
                item["retrieval_score"] = float(item.get("bm25_score", 0.0))
        else:
            candidate_idx = self.store._candidate_indexes(company=company, year=year)
            bm25 = self.store._bm25.search(query, candidate_idx, self.config.warehouse_top_k)
            emb = self.store._embedding_search(query, candidate_idx, self.config.warehouse_top_k)
            chunks = self.store._fuse_results(bm25, emb, self.config.warehouse_top_k)
            for item in chunks:
                item["retrieval_score"] = float(item.get("hybrid_score", 0.0))

        chunks = self._dedupe_chunks(chunks)[: self.config.top_k]
        for rank, chunk in enumerate(chunks, start=1):
            chunk["rank"] = rank
            chunk["topic_id"] = topic_id
            chunk["topic_name"] = topic_name
            chunk["reranker_score"] = float(chunk.get("retrieval_score", 0.0))
            chunk["reranker_model"] = "topic-warehouse-no-indicator-rerank"

        write_table(path, pd.DataFrame(chunks))
        return TopicWarehouse(topic_id, topic_name, chunks, False, perf_counter() - start, 1)

    def _legacy_topic_from_indicator_cache(self, company: str, year: int | None, group: pd.DataFrame) -> list[dict]:
        chunks: list[dict] = []
        for _, indicator in group.iterrows():
            legacy = self.cache_dir / f"{company}_{year}_{indicator.indicator_id}.parquet"
            if not legacy.exists():
                continue
            frame = read_table(legacy)
            if frame.empty:
                continue
            row = frame.iloc[0].to_dict()
            text = str(row.get("evidence", ""))
            if not text:
                continue
            chunks.append(
                {
                    "chunk_id": f"legacy_{company}_{year}_{indicator.indicator_id}",
                    "company": company,
                    "year": year,
                    "source_file": f"{company}_{year}",
                    "page": str(row.get("page_numbers", "")),
                    "text": text,
                    "retrieval_score": float(row.get("retrieval_score", 0.0) or 0.0),
                    "reranker_score": float(row.get("rerank_score", 0.0) or 0.0),
                    "reranker_model": "legacy-indicator-cache-migrated-to-topic",
                }
            )
        return self._dedupe_chunks(chunks)[: self.config.top_k] if chunks else []

    def _row_from_topic(
        self,
        company: str,
        year: int | None,
        indicator: Any,
        topic: TopicWarehouse,
        topic_reuse_count: int,
        warehouse_hits: int,
        warehouse_misses: int,
    ) -> dict:
        chunks = topic.chunks[: self.config.top_k]
        bundle = build_evidence_bundle(str(indicator.indicator_id), chunks)
        evidence = bundle["evidence_bundle"]
        chunk_count = len(bundle["chunk_ids"])
        retrieval_score = max((float(chunk.get("retrieval_score", 0.0)) for chunk in chunks), default=0.0)
        rerank_score = max((float(chunk.get("reranker_score", 0.0)) for chunk in chunks), default=0.0)
        return {
            "company": company,
            "year": year,
            "indicator_id": indicator.indicator_id,
            "pillar": indicator.pillar,
            "indicator_name": indicator.indicator_name_vi,
            "definition": indicator.definition,
            "framework": indicator.framework,
            "retrieval_query": indicator.retrieval_query,
            "evidence": evidence,
            "page_numbers": ",".join(bundle["pages"]),
            "evidence_length": len(evidence),
            "chunk_count": chunk_count,
            "page_count": len(bundle["pages"]),
            "evidence_quality": "LOW_EVIDENCE" if chunk_count < 2 else "OK",
            "retrieval_score": retrieval_score,
            "rerank_score": rerank_score,
            "candidate_count": len(topic.chunks),
            "topic_id": topic.topic_id,
            "topic_name": topic.topic_name,
            "topic_reuse_count": topic_reuse_count,
            "warehouse_hits": warehouse_hits,
            "warehouse_misses": warehouse_misses,
        }

    def _topic_query(self, group: pd.DataFrame) -> str:
        parts = []
        for _, row in group.iterrows():
            parts.extend([str(row.get("topic_query", "")), str(row.get("retrieval_query", "")), str(row.get("keywords_vi", ""))])
        return " ".join(parts)

    def _dedupe_chunks(self, chunks: list[dict]) -> list[dict]:
        seen: set[str] = set()
        output: list[dict] = []
        for chunk in sorted(chunks, key=lambda item: item.get("retrieval_score", 0.0), reverse=True):
            key = str(chunk.get("chunk_id") or chunk.get("text", "")[:160])
            if key in seen:
                continue
            seen.add(key)
            output.append(chunk)
        return output

    def _write_indicator_cache(self, path: Path, row: dict, report_hash: str, topic_id: str) -> None:
        cached = dict(row)
        cached["report_hash"] = report_hash
        cached["topic_id"] = topic_id
        cached["retrieval_config_hash"] = self.config_hash
        write_table(path, pd.DataFrame([cached]))

    def _topic_cache_path(self, report_hash: str, topic_id: str) -> Path:
        safe = topic_id.replace("/", "_").replace("\\", "_")
        return self.warehouse_dir / f"{report_hash}_{safe}_{self.config_hash}.parquet"

    def _indicator_cache_path(self, report_hash: str, indicator_id: str) -> Path:
        safe = indicator_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{report_hash}_{safe}_{self.config_hash}.parquet"

    def _report_hash(self, report_id: str) -> str:
        row = self.manifest.frame.loc[self.manifest.frame["report_id"] == report_id]
        if row.empty:
            return hashlib.sha256(report_id.encode("utf-8")).hexdigest()
        return str(row.iloc[0]["file_hash"])

    def _mark_retrieved(self, report_id: str) -> None:
        row = self.manifest.frame.loc[self.manifest.frame["report_id"] == report_id]
        if row.empty:
            return
        with self._manifest_lock:
            self.manifest.mark(ReportIdentity(report_id, str(row.iloc[0]["file_hash"]), Path("")), "retrieved")


def retrieval_config_hash(config: RetrievalConfig) -> str:
    payload = {
        "top_k": config.top_k,
        "prefetch_k": config.prefetch_k,
        "version": config.version,
        "reranker_model": config.reranker_model,
        "rerank_threshold": config.rerank_threshold,
        "retrieval_mode": config.retrieval_mode,
        "warehouse_top_k": config.warehouse_top_k,
        "topic_cache": True,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
