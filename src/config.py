from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelinePaths:
    root: Path = ROOT
    raw_reports: Path = ROOT / "data" / "raw_reports"
    parsed_reports: Path = ROOT / "data" / "parsed_reports"
    chunks: Path = ROOT / "data" / "chunks"
    embeddings: Path = ROOT / "data" / "embeddings"
    ocr_cache: Path = ROOT / "data" / "ocr_cache"
    indicators: Path = ROOT / "indicators"
    vectorstore: Path = ROOT / "vectorstore" / "faiss"
    indicator_scores: Path = ROOT / "outputs" / "indicator_scores"
    esg_scores: Path = ROOT / "outputs" / "ESG_scores"
    evidence: Path = ROOT / "outputs" / "evidence"
    evidence_dataset: Path = ROOT / "outputs" / "evidence_dataset.csv"
    chatgpt_batches: Path = ROOT / "outputs" / "chatgpt_batches"
    chatgpt_batches_json: Path = ROOT / "outputs" / "chatgpt_batches_json"
    chatgpt_prompt_package: Path = ROOT / "outputs" / "chatgpt_prompt_package"
    chatgpt_batch_manifest: Path = ROOT / "outputs" / "chatgpt_batch_manifest.csv"
    errors: Path = ROOT / "outputs" / "errors.csv"
    review_queue: Path = ROOT / "outputs" / "review_queue.csv"
    indicator_evidence: Path = ROOT / "outputs" / "indicator_evidence.csv"
    retrieval_log: Path = ROOT / "outputs" / "retrieval_log.csv"
    prompt_log: Path = ROOT / "outputs" / "prompt_log.csv"
    model_log: Path = ROOT / "outputs" / "model_log.csv"
    robustness_report: Path = ROOT / "outputs" / "robustness_report.csv"
    progress_manifest: Path = ROOT / "outputs" / "progress_manifest.parquet"
    ocr_manifest: Path = ROOT / "outputs" / "ocr_manifest.parquet"
    ocr_errors: Path = ROOT / "outputs" / "ocr_errors.csv"
    evidence_cache: Path = ROOT / "outputs" / "evidence_cache"
    evidence_warehouse: Path = ROOT / "outputs" / "evidence_warehouse"
    retrieval_runtime: Path = ROOT / "outputs" / "retrieval_runtime.csv"
    warehouse_benchmark: Path = ROOT / "outputs" / "warehouse_benchmark.csv"
    experiments: Path = ROOT / "outputs" / "experiments"
    logs: Path = ROOT / "logs"
    pipeline_log: Path = ROOT / "logs" / "pipeline.log"
    duckdb_path: Path = ROOT / "data" / "esg_pipeline.duckdb"

    def ensure(self) -> None:
        for path in (
            self.raw_reports,
            self.parsed_reports,
            self.chunks,
            self.embeddings,
            self.ocr_cache,
            self.indicators,
            self.vectorstore,
            self.indicator_scores,
            self.esg_scores,
            self.evidence,
            self.chatgpt_batches,
            self.chatgpt_batches_json,
            self.chatgpt_prompt_package,
            self.evidence_cache,
            self.evidence_warehouse,
            self.experiments,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 650
    overlap: int = 100


@dataclass(frozen=True)
class EmbeddingConfig:
    dimensions: int = 1024


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    prefetch_k: int = 20
    version: str = "hybrid-bm25-embedding-bge-reranker-v1"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_threshold: float = 0.95
    retrieval_mode: str = "hybrid-rerank"
    warehouse_top_k: int = 80


@dataclass(frozen=True)
class CodebookConfig:
    filename: str = "esg_codebook_vn_50_indicators.csv"
    required_columns: tuple[str, ...] = (
        "indicator_id",
        "pillar",
        "category_vi",
        "indicator_name_vi",
        "framework",
        "retrieval_query",
        "keywords_vi",
        "score_0",
        "score_1",
        "score_2",
        "score_3",
        "evidence_requirement",
    )


@dataclass(frozen=True)
class PromptConfig:
    prompt_dir: Path = ROOT / "prompts"
    filename: str = "esg_scoring_prompt.txt"
    version: str = "v1.0"
