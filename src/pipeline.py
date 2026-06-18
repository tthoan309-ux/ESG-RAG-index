from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone

import pandas as pd

from .batch_builder import build_chatgpt_batches
from .benchmarking import per_hour
from .cache_manager import CacheManager
from .config import ChunkingConfig, EmbeddingConfig, PipelinePaths, RetrievalConfig
from .corpus_manager import CorpusConfig, CorpusManager
from .duckdb_backend import DuckDBBackend
from .embedding_manager import EmbeddingManager
from .error_manager import ErrorRecorder
from .export_evidence import load_codebook
from .experiment_tracker import write_experiment
from .logging_config import configure_logging
from .manifest_manager import ManifestManager
from .ocr_manager import default_ocr_workers
from .reranker import BGEReranker
from .retrieval_manager import RetrievalManager
from .retriever import VectorStore, save_vectorstore


def firm_years(chunks: list[dict]) -> list[tuple[str, int | None]]:
    return sorted({(chunk["company"], chunk["year"]) for chunk in chunks}, key=lambda item: (item[0], item[1] or 0))


def run_pipeline(
    input_dir: Path | None = None,
    limit: int | None = None,
    top_k: int = 5,
    batch_size: int = 100,
    include_glob: str = "*",
    force_ocr: bool = False,
    ocr_language: str = "vie+eng",
    ocr_min_chars: int = 500,
    ocr_min_words_per_page: float = 20.0,
    ocr_workers: int | None = None,
    ocr_timeout: int = 60,
    rebuild_parsed: bool = False,
    rebuild_chunks: bool = False,
    rebuild_embeddings: bool = False,
    rebuild_retrieval: bool = False,
    incremental: bool = False,
    resume: bool = False,
    embedding_batch_size: int = 5000,
    retrieval_workers: int = 1,
    retrieval_mode: str = "hybrid-rerank",
    rerank_threshold: float = 0.95,
    warehouse_top_k: int = 80,
) -> pd.DataFrame:
    total_start = perf_counter()
    paths = PipelinePaths(raw_reports=input_dir or PipelinePaths().raw_reports)
    paths.ensure()
    configure_logging(paths.pipeline_log)
    logger = logging.getLogger(__name__)
    logger.info("Starting ESG corpus pipeline")

    errors = ErrorRecorder()
    cache = CacheManager(paths.root)
    cache.ensure()
    manifest_manager = ManifestManager(paths.progress_manifest)

    corpus_config = CorpusConfig(
        include_glob=include_glob,
        limit=limit,
        force_ocr=force_ocr,
        ocr_language=ocr_language,
        ocr_threshold_chars=ocr_min_chars,
        ocr_threshold_words_per_page=ocr_min_words_per_page,
        ocr_workers=ocr_workers,
        ocr_timeout_seconds=ocr_timeout,
        rebuild_parsed=rebuild_parsed,
        rebuild_chunks=rebuild_chunks,
        incremental=incremental,
        resume=resume,
    )

    corpus = CorpusManager(paths.raw_reports, cache, manifest_manager, errors)
    chunk_config = ChunkingConfig()
    embedding_config = EmbeddingConfig()
    corpus_result = corpus.process(corpus_config, chunk_config)
    if not corpus_result.chunks:
        errors.write_csv(paths.errors)
        raise RuntimeError("No chunks were produced from the corpus. Check outputs/errors.csv and OCR dependencies.")

    embedding_start = perf_counter()
    embedding_manager = EmbeddingManager(cache, manifest_manager, embedding_config, batch_size=embedding_batch_size)
    embedding_result = embedding_manager.embed(corpus_result.chunks, rebuild=rebuild_embeddings)
    embedding_seconds = perf_counter() - embedding_start

    maybe_save_vectorstore(paths, embedding_result.embeddings, corpus_result.chunks)
    store = VectorStore(embedding_result.embeddings, corpus_result.chunks, embedding_manager.embedder)

    retrieval_start = perf_counter()
    codebook = load_codebook(paths.indicators)
    retrieval_config = RetrievalConfig(
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        rerank_threshold=rerank_threshold,
        warehouse_top_k=warehouse_top_k,
    )
    reranker = BGEReranker()
    retrieval_result = RetrievalManager(
        store,
        reranker,
        retrieval_config,
        manifest_manager,
        cache_dir=paths.evidence_cache,
        rebuild=rebuild_retrieval,
        workers=retrieval_workers,
    ).export_evidence(
        codebook=codebook,
        firm_years=firm_years(corpus_result.chunks),
        output_path=paths.evidence_dataset,
    )
    retrieval_seconds = perf_counter() - retrieval_start

    batch_manifest = build_chatgpt_batches(
        evidence_dataset_path=paths.evidence_dataset,
        output_dir=paths.chatgpt_batches,
        json_output_dir=paths.chatgpt_batches_json,
        prompt_package_dir=paths.chatgpt_prompt_package,
        manifest_path=paths.chatgpt_batch_manifest,
        batch_size=batch_size,
    )

    total_seconds = perf_counter() - total_start
    experiment = write_experiment(paths.experiments, embedding_config, retrieval_config, chunk_config, total_seconds)
    errors.write_csv(paths.errors)
    duckdb = DuckDBBackend(paths.duckdb_path)
    duckdb.write_table("reports", pd.DataFrame([record.__dict__ for record in corpus_result.records]))
    duckdb.write_table("chunks", pd.DataFrame(corpus_result.chunks))
    duckdb.write_table(
        "embeddings_metadata",
        pd.DataFrame(
            [
                {
                    "embedding_count": embedding_result.embedding_count,
                    "embedding_reused": embedding_result.reused_count,
                    "embedding_new": embedding_result.new_count,
                    "embedding_path": str(paths.embeddings / "embeddings.npy"),
                }
            ]
        ),
    )
    duckdb.write_table("retrieval_results", retrieval_result.evidence)
    manifest = build_manifest(
        paths=paths,
        corpus_result=corpus_result,
        embedding_count=embedding_result.embedding_count,
        embedding_reused=embedding_result.reused_count,
        embedding_new=embedding_result.new_count,
        retrieval_count=retrieval_result.retrieval_count,
        retrieval_reused=retrieval_result.reused_count,
        experiment=experiment,
        batch_count=len(batch_manifest),
        embedding_seconds=embedding_seconds,
        retrieval_seconds=retrieval_seconds,
        total_seconds=total_seconds,
    )
    (paths.root / "outputs" / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Completed ESG corpus pipeline in %.3fs", total_seconds)
    print_benchmark(manifest)
    return retrieval_result.evidence


def build_manifest(
    paths: PipelinePaths,
    corpus_result,
    embedding_count: int,
    embedding_reused: int,
    embedding_new: int,
    retrieval_count: int,
    retrieval_reused: int,
    experiment: dict,
    batch_count: int,
    embedding_seconds: float,
    retrieval_seconds: float,
    total_seconds: float,
) -> dict:
    reports_per_hour = per_hour(corpus_result.reports_processed, total_seconds)
    pages_per_hour = per_hour(corpus_result.pages_processed, total_seconds)
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "reports_found": corpus_result.reports_found,
        "reports_processed": corpus_result.reports_processed,
        "reports_skipped": corpus_result.reports_skipped,
        "ocr_reports": corpus_result.ocr_reports,
        "failed_reports": corpus_result.failed_reports,
        "chunk_count": len(corpus_result.chunks),
        "embedding_count": embedding_count,
        "embedding_reused": embedding_reused,
        "embedding_new": embedding_new,
        "retrieval_count": retrieval_count,
        "retrieval_reused": retrieval_reused,
        "runtime_seconds": round(total_seconds, 3),
        "parsing_time_seconds": round(corpus_result.parsing_seconds, 3),
        "ocr_time_seconds": round(corpus_result.ocr_seconds, 3),
        "ocr_runtime": round(corpus_result.ocr_seconds, 3),
        "ocr_pages": corpus_result.ocr_pages,
        "ocr_failed_pages": corpus_result.ocr_failed_pages,
        "ocr_pages_per_hour": round(corpus_result.ocr_pages_per_hour, 3),
        "ocr_failure_rate": round(corpus_result.ocr_failure_rate, 6),
        "chunking_time_seconds": round(corpus_result.chunking_seconds, 3),
        "embedding_time_seconds": round(embedding_seconds, 3),
        "retrieval_time_seconds": round(retrieval_seconds, 3),
        "reports_per_hour": round(reports_per_hour, 3),
        "pages_per_hour": round(pages_per_hour, 3),
        "parsed_files": [str(path) for path in corpus_result.parsed_files],
        "chunk_files": [str(path) for path in corpus_result.chunk_files],
        "embeddings": str(paths.embeddings / "embeddings.npy"),
        "embedding_manifest": str(paths.embeddings / "embedding_manifest.json"),
        "vectorstore": str(paths.vectorstore / "vectorstore.npz"),
        "evidence_dataset": str(paths.evidence_dataset),
        "chatgpt_batches": str(paths.chatgpt_batches),
        "chatgpt_batches_json": str(paths.chatgpt_batches_json),
        "chatgpt_prompt_package": str(paths.chatgpt_prompt_package),
        "chatgpt_batch_manifest": str(paths.chatgpt_batch_manifest),
        "errors": str(paths.errors),
        "ocr_manifest": str(paths.ocr_manifest),
        "ocr_errors": str(paths.ocr_errors),
        "progress_manifest": str(paths.progress_manifest),
        "evidence_cache": str(paths.evidence_cache),
        "evidence_warehouse": str(paths.evidence_warehouse),
        "retrieval_runtime": str(paths.retrieval_runtime),
        "pipeline_log": str(paths.pipeline_log),
        "duckdb": str(paths.duckdb_path),
        "experiment": experiment,
        "batch_count": batch_count,
    }


def maybe_save_vectorstore(paths: PipelinePaths, embeddings, chunks: list[dict]) -> None:
    vectorstore_path = paths.vectorstore / "vectorstore.npz"
    manifest_path = paths.vectorstore / "vectorstore_manifest.json"
    manifest = {
        "chunk_count": len(chunks),
        "embedding_count": len(embeddings),
        "chunk_ids": [str(chunk.get("chunk_id", "")) for chunk in chunks],
    }
    if vectorstore_path.exists() and manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing == manifest:
                return
        except json.JSONDecodeError:
            pass
    save_vectorstore(vectorstore_path, embeddings, chunks)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def print_benchmark(manifest: dict) -> None:
    print("\nPipeline benchmark")
    print(f"Parsing Time:   {manifest['parsing_time_seconds']}s")
    print(f"OCR Time:       {manifest['ocr_time_seconds']}s")
    print(f"OCR Pages:      {manifest['ocr_pages']} pages ({manifest['ocr_pages_per_hour']} pages/hour)")
    print(f"OCR Failures:   {manifest['ocr_failed_pages']} pages ({manifest['ocr_failure_rate']})")
    print(f"Chunking Time:  {manifest['chunking_time_seconds']}s")
    print(f"Embedding Time: {manifest['embedding_time_seconds']}s")
    print(f"Retrieval Time: {manifest['retrieval_time_seconds']}s")
    print(f"Total Time:     {manifest['runtime_seconds']}s")
    print(f"Throughput:     {manifest['reports_per_hour']} reports/hour")
    print(f"Page Rate:      {manifest['pages_per_hour']} pages/hour")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the corpus-scale ESG-RAG-Index evidence pipeline.")
    parser.add_argument("--input-dir", type=Path, default=PipelinePaths().raw_reports)
    parser.add_argument("--include-glob", default="*", help="Process files matching this glob. Default processes all raw reports.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of reports for smoke tests.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=100, help="Indicators per ChatGPT Plus upload batch.")
    parser.add_argument("--ocr", action="store_true", help="Force OCR for PDFs instead of using automatic fallback only.")
    parser.add_argument("--ocr-language", default="vie+eng")
    parser.add_argument("--ocr-min-chars", type=int, default=500)
    parser.add_argument("--ocr-min-words-per-page", type=float, default=20.0)
    parser.add_argument("--ocr-workers", type=int, default=default_ocr_workers())
    parser.add_argument("--ocr-timeout", type=int, default=60, help="Per-page OCR timeout in seconds. Timed-out pages are skipped and cached.")
    parser.add_argument("--rebuild-parsed", action="store_true")
    parser.add_argument("--rebuild-chunks", action="store_true")
    parser.add_argument("--rebuild-embeddings", action="store_true")
    parser.add_argument("--rebuild-retrieval", action="store_true")
    parser.add_argument("--incremental", action="store_true", help="Reuse prior artifacts and process only changed/new reports where possible.")
    parser.add_argument("--resume", action="store_true", help="Resume from progress manifest and skip completed stages.")
    parser.add_argument("--embedding-batch-size", type=int, default=5000)
    parser.add_argument("--retrieval-workers", type=int, default=1)
    parser.add_argument("--retrieval-mode", choices=["bm25", "hybrid", "hybrid-rerank"], default="hybrid-rerank")
    parser.add_argument("--rerank-threshold", type=float, default=0.95)
    parser.add_argument("--warehouse-top-k", type=int, default=80)
    args = parser.parse_args()

    evidence = run_pipeline(
        input_dir=args.input_dir,
        limit=args.limit,
        top_k=args.top_k,
        batch_size=args.batch_size,
        include_glob=args.include_glob,
        force_ocr=args.ocr,
        ocr_language=args.ocr_language,
        ocr_min_chars=args.ocr_min_chars,
        ocr_min_words_per_page=args.ocr_min_words_per_page,
        ocr_workers=args.ocr_workers,
        ocr_timeout=args.ocr_timeout,
        rebuild_parsed=args.rebuild_parsed,
        rebuild_chunks=args.rebuild_chunks,
        rebuild_embeddings=args.rebuild_embeddings,
        rebuild_retrieval=args.rebuild_retrieval,
        incremental=args.incremental,
        resume=args.resume,
        embedding_batch_size=args.embedding_batch_size,
        retrieval_workers=args.retrieval_workers,
        retrieval_mode=args.retrieval_mode,
        rerank_threshold=args.rerank_threshold,
        warehouse_top_k=args.warehouse_top_k,
    )
    print(f"\nWrote {len(evidence)} evidence records to {PipelinePaths().evidence_dataset}")


if __name__ == "__main__":
    main()
