from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd

from .cache_manager import CacheManager
from .config import EmbeddingConfig, PipelinePaths, RetrievalConfig
from .embedder import HashingEmbedder
from .financial_pages import FinancialPageParser
from .financial_retriever import FinancialRetriever
from .manifest_manager import ManifestManager
from .pipeline import firm_years
from .retriever import VectorStore
from .storage import read_table


def run_financial_engine(
    top_k: int = 8,
    warehouse_top_k: int = 40,
    rebuild_financial: bool = False,
    rebuild_financial_pages: bool = False,
    financial_pages_only: bool = True,
    output_csv: Path | None = None,
    output_parquet: Path | None = None,
) -> pd.DataFrame:
    paths = PipelinePaths()
    cache = CacheManager(paths.root)
    chunks_path = cache.combined_chunks_path
    embeddings_path = cache.embeddings_path
    if not chunks_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            "Financial engine needs cached chunks and embeddings. Run `python -m src.pipeline --resume` first."
        )

    embedder = HashingEmbedder(EmbeddingConfig())
    if financial_pages_only:
        corpus = FinancialPageParser(paths=paths, embedder=embedder).build(rebuild=rebuild_financial_pages)
        chunks = corpus.chunks
        store = corpus.store
    else:
        chunks = read_table(chunks_path).to_dict(orient="records")
        embeddings = cache.load_embeddings()
        if embeddings is None:
            raise FileNotFoundError(f"Embedding cache not found: {embeddings_path}")
        store = VectorStore(embeddings, chunks, embedder)
    manifest = ManifestManager(paths.progress_manifest)
    config = RetrievalConfig(top_k=top_k, warehouse_top_k=warehouse_top_k)
    result = FinancialRetriever(
        store=store,
        manifest=manifest,
        config=config,
        rebuild=rebuild_financial,
    ).export_financials(
        firm_years=firm_years(chunks),
        csv_path=output_csv,
        parquet_path=output_parquet,
    )
    return result.financial


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract financial variables from cached annual-report chunks.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warehouse-top-k", type=int, default=40)
    parser.add_argument("--rebuild-financial", action="store_true")
    parser.add_argument("--rebuild-financial-pages", action="store_true")
    parser.add_argument("--all-pages", action="store_true", help="Use the full report chunk corpus instead of financial-pages-only retrieval.")
    parser.add_argument("--output-csv", type=Path, default=PipelinePaths().financial_dataset_csv)
    parser.add_argument("--output-parquet", type=Path, default=PipelinePaths().financial_dataset_parquet)
    args = parser.parse_args()

    start = perf_counter()
    financial = run_financial_engine(
        top_k=args.top_k,
        warehouse_top_k=args.warehouse_top_k,
        rebuild_financial=args.rebuild_financial,
        rebuild_financial_pages=args.rebuild_financial_pages,
        financial_pages_only=not args.all_pages,
        output_csv=args.output_csv,
        output_parquet=args.output_parquet,
    )
    seconds = perf_counter() - start
    extracted = int(financial["value"].notna().sum()) if "value" in financial.columns else 0
    print(f"Wrote {len(financial)} financial rows to {args.output_csv} and {args.output_parquet}")
    print(f"Extracted {extracted} non-null values in {seconds:.3f}s")


if __name__ == "__main__":
    main()
