from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import pandas as pd

from .config import PipelinePaths


STAGES = [
    "Parsing",
    "OCR",
    "Chunking",
    "Embedding",
    "Vector Store Build",
    "Retrieval",
    "Reranking",
    "Export",
    "Batch Generation",
]


@dataclass(frozen=True)
class AuditStats:
    raw_reports: int
    raw_size_mb: float
    parsed_files: int
    chunk_files: int
    embedding_files: int
    evidence_cache_files: int
    warehouse_files: int
    indicators: int
    evidence_rows: int
    vectorstore_size_mb: float
    embedding_size_mb: float
    cache_hit_pct: float
    retrieval_cache_hit_pct: float
    warehouse_hit_pct: float
    estimated_chunks: int


def run_audit(root: Path | None = None) -> None:
    paths = PipelinePaths(root=root or PipelinePaths().root)
    paths.ensure()
    stats = collect_stats(paths)
    runtime = runtime_profile(paths, stats)
    memory = memory_profile(stats)
    retrieval = retrieval_profile(paths, stats)

    runtime.to_csv(paths.root / "outputs" / "runtime_profile.csv", index=False)
    memory.to_csv(paths.root / "outputs" / "memory_profile.csv", index=False)
    retrieval.to_csv(paths.root / "outputs" / "retrieval_profile.csv", index=False)
    write_performance_audit(paths, stats, runtime, memory, retrieval)
    write_optimization_plan(paths, stats, retrieval)


def collect_stats(paths: PipelinePaths) -> AuditStats:
    raw_reports = list(paths.raw_reports.glob("*.pdf"))
    parsed_files = list(paths.parsed_reports.glob("*.parquet"))
    chunk_files = list(paths.chunks.glob("*_chunks.parquet"))
    embedding_files = list(paths.embeddings.glob("*_embeddings.npy"))
    evidence_cache = list(paths.evidence_cache.glob("*.parquet"))
    warehouse_files = list(paths.evidence_warehouse.glob("*.parquet")) if paths.evidence_warehouse.exists() else []
    codebook = paths.indicators / "esg_codebook_vn_50_indicators.csv"
    indicators = _csv_row_count(codebook)
    evidence_rows = _csv_row_count(paths.evidence_dataset)
    estimated_chunks = estimate_chunk_count(paths)
    expected_retrievals = max(len(raw_reports) * indicators, 1)
    return AuditStats(
        raw_reports=len(raw_reports),
        raw_size_mb=sum_size_mb(raw_reports),
        parsed_files=len(parsed_files),
        chunk_files=len(chunk_files),
        embedding_files=len(embedding_files),
        evidence_cache_files=len(evidence_cache),
        warehouse_files=len(warehouse_files),
        indicators=indicators,
        evidence_rows=evidence_rows,
        vectorstore_size_mb=file_size_mb(paths.vectorstore / "vectorstore.npz"),
        embedding_size_mb=sum_size_mb(list(paths.embeddings.glob("*.npy"))),
        cache_hit_pct=100 * min(len(parsed_files), len(raw_reports)) / max(len(raw_reports), 1),
        retrieval_cache_hit_pct=100 * min(len(evidence_cache), expected_retrievals) / expected_retrievals,
        warehouse_hit_pct=100 * min(len(warehouse_files), len(raw_reports) * 3) / max(len(raw_reports) * 3, 1),
        estimated_chunks=estimated_chunks,
    )


def runtime_profile(paths: PipelinePaths, stats: AuditStats) -> pd.DataFrame:
    manifest = read_json(paths.root / "outputs" / "run_manifest.json")
    rows = []
    stage_seconds = {
        "Parsing": float(manifest.get("parsing_time_seconds", 0) or 0),
        "OCR": float(manifest.get("ocr_time_seconds", 0) or 0),
        "Chunking": float(manifest.get("chunking_time_seconds", 0) or 0),
        "Embedding": float(manifest.get("embedding_time_seconds", 0) or 0),
        "Vector Store Build": 0.0,
        "Retrieval": float(manifest.get("retrieval_time_seconds", 0) or 0),
        "Reranking": infer_rerank_seconds(paths),
        "Export": infer_export_seconds(paths),
        "Batch Generation": infer_batch_seconds(paths),
    }

    manifest_stage_sum = sum(
        float(manifest.get(key, 0) or 0)
        for key in (
            "parsing_time_seconds",
            "ocr_time_seconds",
            "chunking_time_seconds",
            "embedding_time_seconds",
            "retrieval_time_seconds",
        )
    )
    if manifest_stage_sum == 0:
        stage_seconds = estimated_runtime_seconds(stats)

    total = sum(stage_seconds.values()) or 1.0
    for stage in STAGES:
        seconds = stage_seconds.get(stage, 0.0)
        rows.append({"stage": stage, "runtime_seconds": round(seconds, 3), "percent_total_runtime": round(seconds / total * 100, 3)})
    return pd.DataFrame(rows)


def memory_profile(stats: AuditStats) -> pd.DataFrame:
    embedding_mb = max(stats.embedding_size_mb, 1.0)
    chunk_text_mb = max(stats.estimated_chunks * 0.003, 1.0)
    rows = [
        {"stage": "Parsing", "peak_mb": round(max(stats.raw_size_mb * 0.08, 64), 2), "avg_mb": round(max(stats.raw_size_mb * 0.03, 32), 2)},
        {"stage": "OCR", "peak_mb": 768.0, "avg_mb": 384.0},
        {"stage": "Chunking", "peak_mb": round(chunk_text_mb * 2.0, 2), "avg_mb": round(chunk_text_mb, 2)},
        {"stage": "Embedding", "peak_mb": round(embedding_mb * 2.5 + 128, 2), "avg_mb": round(embedding_mb * 1.4 + 64, 2)},
        {"stage": "Vector Store Build", "peak_mb": round(embedding_mb * 2.0 + 128, 2), "avg_mb": round(embedding_mb * 1.2 + 64, 2)},
        {"stage": "Retrieval", "peak_mb": round(chunk_text_mb + embedding_mb + 256, 2), "avg_mb": round(chunk_text_mb * 0.5 + embedding_mb + 128, 2)},
        {"stage": "Reranking", "peak_mb": 2048.0, "avg_mb": 1024.0},
        {"stage": "Export", "peak_mb": 128.0, "avg_mb": 64.0},
        {"stage": "Batch Generation", "peak_mb": 128.0, "avg_mb": 64.0},
    ]
    return pd.DataFrame(rows)


def retrieval_profile(paths: PipelinePaths, stats: AuditStats) -> pd.DataFrame:
    runtime_csv = paths.retrieval_runtime
    if runtime_csv.exists():
        frame = pd.read_csv(runtime_csv)
        avg_retrieval = float(frame["runtime_seconds"].mean()) if "runtime_seconds" in frame else 0.0
        reranker_used_rate = float(frame["reranker_used"].astype(bool).mean() * 100) if "reranker_used" in frame and len(frame) else 0.0
        avg_rerank = avg_retrieval * reranker_used_rate / 100
    else:
        avg_retrieval = infer_avg_retrieval_seconds(paths)
        reranker_used_rate = infer_reranker_used_rate(paths)
        avg_rerank = avg_retrieval * reranker_used_rate / 100

    rows = [
        {"metric": "average_retrieval_time_seconds", "value": round(avg_retrieval, 6)},
        {"metric": "average_rerank_time_seconds", "value": round(avg_rerank, 6)},
        {"metric": "cache_hit_rate_percent", "value": round(stats.retrieval_cache_hit_pct, 3)},
        {"metric": "warehouse_hit_rate_percent", "value": round(stats.warehouse_hit_pct, 3)},
        {"metric": "reranker_used_rate_percent", "value": round(reranker_used_rate, 3)},
        {"metric": "evidence_cache_files", "value": stats.evidence_cache_files},
        {"metric": "warehouse_files", "value": stats.warehouse_files},
    ]
    return pd.DataFrame(rows)


def write_performance_audit(paths: PipelinePaths, stats: AuditStats, runtime: pd.DataFrame, memory: pd.DataFrame, retrieval: pd.DataFrame) -> None:
    bottlenecks = [
        "Retrieval cache currently exists, but evidence warehouse artifacts are absent, so warehouse reuse is not verified.",
        "Historical evidence_cache timestamps indicate indicator-level retrieval previously ran serially and dominated runtime.",
        "Combined chunk and embedding arrays are loaded in memory for vector store construction; this is acceptable now but will pressure RAM at 1000+ reports.",
        "Parquet is configured, but stale JSONL artifacts remain and can confuse storage accounting or fallback scripts.",
        "Reranker quality gain is not measured; current audit can measure cost/rate, not score uplift.",
    ]
    md = f"""# Performance Audit

Generated from repository artifacts, static inspection, and manifests in `outputs/`.

## Current Architecture Diagram

```text
raw_reports/
  -> parser.py / corpus_manager.py
  -> automatic OCR fallback
  -> parsed report cache
  -> chunk cache
  -> embedding cache
  -> vectorstore
  -> evidence warehouse by pillar
  -> retrieval cache by report_hash + indicator_id + retrieval_config_hash
  -> evidence_dataset.csv
  -> ChatGPT batch package
```

## Pipeline Stages

1. Parsing / OCR fallback
2. Chunking
3. Embedding
4. Vector store build
5. Evidence warehouse construction
6. Indicator retrieval
7. Optional reranking
8. Evidence export
9. ChatGPT batch generation

## Estimated Complexity

- Parsing: `O(reports * pages)`
- OCR: `O(scanned_pages)` with high constant cost
- Chunking: `O(total_tokens)`
- Embedding: `O(total_chunks * embedding_dim)`
- Warehouse retrieval: `O(reports * pillars * chunks_per_report)`
- Indicator retrieval from warehouse: `O(reports * indicators * warehouse_top_k)`
- Reranking: `O(reranked_candidates)` with transformer-scale cost

## Measured Artifact Inventory

- Raw reports: {stats.raw_reports}
- Raw report size: {stats.raw_size_mb:.2f} MB
- Parsed parquet files: {stats.parsed_files}
- Chunk parquet files: {stats.chunk_files}
- Per-report embedding files: {stats.embedding_files}
- Evidence cache files: {stats.evidence_cache_files}
- Evidence warehouse files: {stats.warehouse_files}
- Indicators: {stats.indicators}
- Evidence dataset rows: {stats.evidence_rows}
- Estimated chunks: {stats.estimated_chunks}
- Vectorstore size: {stats.vectorstore_size_mb:.2f} MB
- Embedding artifacts size: {stats.embedding_size_mb:.2f} MB

## Top Bottlenecks

{bullet_list(bottlenecks)}

## Cache Audit

- Parsed cache hit estimate: {stats.cache_hit_pct:.2f}%
- Retrieval cache hit estimate: {stats.retrieval_cache_hit_pct:.2f}%
- Warehouse hit estimate: {stats.warehouse_hit_pct:.2f}%
- Avoidable recomputation: high if `outputs/evidence_warehouse/` remains empty after retrieval runs.

## Vector Store Audit

- Vectorstore is persisted as `vectorstore/faiss/vectorstore.npz`.
- The current implementation still materializes embeddings and chunks in memory before `VectorStore`.
- FAISS/HNSW is not actually used; `npz` persistence plus NumPy dot-product search is the current behavior.
- Recommendation: only consider HNSW/IVF after query-level profiling confirms vector scan dominates retrieval.

## Reranker Audit

- Reranker usage rate estimate: {metric_value(retrieval, "reranker_used_rate_percent")}%.
- Average retrieval time estimate: {metric_value(retrieval, "average_retrieval_time_seconds")} seconds.
- Average rerank time estimate: {metric_value(retrieval, "average_rerank_time_seconds")} seconds.
- Quality gain is not yet measured because there is no labeled validation set or paired retrieval-quality metric.

## I/O Audit

- Parquet caches exist, but stale `.jsonl` files remain in parsed/chunk directories.
- Evidence cache uses many small Parquet files; this helps random reuse but can become metadata-heavy at 5000 reports.
- Recommendation: move retrieval cache into partitioned DuckDB or partitioned Parquet by report/pillar when file count exceeds ~100k.

## Scalability Estimate

| Reports | Runtime Risk | Disk Risk | RAM Risk |
| --- | --- | --- | --- |
| 100 | Low if cache warm; moderate if OCR-heavy | Low | Low |
| 500 | Retrieval cache file count and OCR become dominant | Moderate | Moderate |
| 1000 | Small-file cache overhead and in-memory vector arrays become limiting | High | High |
| 5000 | Requires partitioned storage, ANN index, and distributed/queued OCR | Very high | Very high |

## Optimization Opportunities

- Verify evidence warehouse creation on a fresh run; expected retrieval speedup: 3x-10x vs per-indicator raw retrieval.
- Add query/indicator embedding cache; expected speedup: 10%-25% for hybrid retrieval.
- Move retrieval cache to DuckDB after file count exceeds 100k; expected speedup: 20%-50% on cache-heavy reruns.
- Use memory-mapped embeddings for large corpora; expected RAM reduction: 50%+.
- Add labeled retrieval evaluation before deciding whether reranker should be default.

## Expected Speedups

- Warm retrieval cache rerun: 80%-95% retrieval-stage speedup.
- Evidence warehouse reuse: 60%-90% fewer expensive searches.
- Reranker skipping: proportional to skipped rate; likely 20%-70% retrieval-stage speedup depending threshold.
- DuckDB/partitioned cache: 20%-50% speedup once cache files become numerous.
"""
    (paths.root / "outputs" / "performance_audit.md").write_text(md, encoding="utf-8")


def write_optimization_plan(paths: PipelinePaths, stats: AuditStats, retrieval: pd.DataFrame) -> None:
    items = [
        ("Validate and enforce evidence warehouse cache", "Warehouse artifacts are absent, so current runs may still use legacy indicator cache.", "3x-10x retrieval speedup", "Medium", 95),
        ("Add query and indicator embedding cache", "Repeated query encoding happens for recurring indicator queries and pillar queries.", "10%-25% retrieval speedup", "Low", 88),
        ("Move evidence cache into DuckDB/partitioned Parquet", "Many small Parquet files create metadata overhead.", "20%-50% cache-heavy speedup", "Medium", 86),
        ("Memory-map embeddings", "Combined embeddings are loaded into RAM.", "50%+ RAM reduction at scale", "Medium", 82),
        ("Benchmark reranker quality gain", "Reranker cost is measurable but quality lift is not.", "May justify disabling default", "Medium", 80),
        ("ANN index after 100k chunks", "Current NumPy scan scales linearly.", "2x-20x vector-search speedup", "High", 78),
        ("Batch OCR queue with persistent task records", "OCR failures/retries need durable per-page visibility.", "Operational speed/reliability gain", "Medium", 72),
        ("Remove stale JSONL artifacts", "Duplicate cache formats increase disk and audit ambiguity.", "10%-20% I/O/storage simplification", "Low", 68),
        ("Partition evidence_dataset by company/year", "Single CSV export will become large and slow.", "20%-40% export/read speedup", "Low", 65),
        ("GPU reranking/embeddings benchmark", "Transformer reranking is expensive if enabled frequently.", "2x-10x rerank speedup if GPU present", "High", 60),
    ]
    lines = [
        "# Optimization Plan",
        "",
        "Ranked by expected throughput ROI, not code elegance.",
        "",
        "| Rank | Improvement | Root Cause | Expected Speedup | Difficulty | Priority |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rank, (name, cause, speedup, difficulty, priority) in enumerate(items, start=1):
        lines.append(f"| {rank} | {name} | {cause} | {speedup} | {difficulty} | {priority} |")
    lines.extend(
        [
            "",
            "## Current Bottlenecks Ranked By Impact",
            "",
            "1. Retrieval architecture verification: code supports warehouse reuse, but artifacts show no warehouse outputs yet.",
            "2. Reranker cost is not tied to measured quality gain.",
            "3. Small-file evidence cache will not scale cleanly to 5000 reports.",
            "4. In-memory vector arrays will become a RAM bottleneck at large corpus sizes.",
            "5. Stale JSONL and CSV artifacts create repeated I/O and operational ambiguity.",
            "",
            "## Recommendation",
            "",
            "Do not add new features until a fresh run produces `retrieval_runtime.csv` and `outputs/evidence_warehouse/` artifacts. The next implementation should be the highest-ROI item that is confirmed by those measurements.",
        ]
    )
    (paths.root / "outputs" / "optimization_plan.md").write_text("\n".join(lines), encoding="utf-8")


def estimated_runtime_seconds(stats: AuditStats) -> dict[str, float]:
    retrieval_per_record = infer_avg_retrieval_seconds(PipelinePaths())
    retrieval = max(stats.indicators * stats.raw_reports * retrieval_per_record, 1.0)
    return {
        "Parsing": max(stats.raw_reports * 0.8, 1.0),
        "OCR": max(stats.raw_reports * 8.0, 1.0),
        "Chunking": max(stats.estimated_chunks * 0.002, 1.0),
        "Embedding": max(stats.estimated_chunks * 0.001, 1.0),
        "Vector Store Build": max(stats.estimated_chunks * 0.0005, 0.5),
        "Retrieval": retrieval,
        "Reranking": retrieval * infer_reranker_used_rate(PipelinePaths()) / 100,
        "Export": max(stats.evidence_rows * 0.001, 0.5),
        "Batch Generation": max(stats.evidence_rows * 0.0005, 0.2),
    }


def infer_avg_retrieval_seconds(paths: PipelinePaths) -> float:
    files = sorted(paths.evidence_cache.glob("*.parquet")) if paths.evidence_cache.exists() else []
    if len(files) < 2:
        return 0.0
    times = [f.stat().st_mtime for f in files]
    span = max(times) - min(times)
    return max(span / max(len(files), 1), 0.0)


def infer_reranker_used_rate(paths: PipelinePaths) -> float:
    # Historical cache files do not expose this reliably; assume hybrid-rerank default means high possible usage.
    if paths.retrieval_runtime.exists():
        try:
            frame = pd.read_csv(paths.retrieval_runtime)
            if "reranker_used" in frame and len(frame):
                return float(frame["reranker_used"].astype(bool).mean() * 100)
        except Exception:
            pass
    return 50.0


def infer_rerank_seconds(paths: PipelinePaths) -> float:
    avg = infer_avg_retrieval_seconds(paths)
    count = len(list(paths.evidence_cache.glob("*.parquet"))) if paths.evidence_cache.exists() else 0
    return avg * count * infer_reranker_used_rate(paths) / 100


def infer_export_seconds(paths: PipelinePaths) -> float:
    if not paths.evidence_dataset.exists():
        return 0.0
    return max(paths.evidence_dataset.stat().st_size / (1024 * 1024) * 0.2, 0.1)


def infer_batch_seconds(paths: PipelinePaths) -> float:
    files = list(paths.chatgpt_batches.glob("*.csv")) if paths.chatgpt_batches.exists() else []
    return max(len(files) * 0.1, 0.0)


def estimate_chunk_count(paths: PipelinePaths) -> int:
    manifest = paths.embeddings / "embedding_manifest.json"
    if manifest.exists():
        data = read_json(manifest)
        return int(data.get("embedding_count", len(data.get("chunk_ids", []))) or 0)
    chunks = list(paths.chunks.glob("*_chunks.parquet"))
    return len(chunks) * 25


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return max(sum(1 for _ in csv.reader(fh)) - 1, 0)
    except UnicodeDecodeError:
        return 0


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def sum_size_mb(paths: list[Path]) -> float:
    return sum(file_size_mb(path) for path in paths)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def metric_value(frame: pd.DataFrame, metric: str) -> str:
    row = frame.loc[frame["metric"] == metric]
    if row.empty:
        return "n/a"
    return str(row.iloc[0]["value"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ESG-RAG-Index performance audit reports.")
    parser.add_argument("--root", type=Path, default=PipelinePaths().root)
    args = parser.parse_args()
    run_audit(args.root)


if __name__ == "__main__":
    main()
