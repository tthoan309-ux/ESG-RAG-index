from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import CodebookConfig, PipelinePaths, RetrievalConfig
from .evidence_builder import build_evidence_bundle
from .reranker import BGEReranker
from .retriever import VectorStore


EVIDENCE_COLUMNS = [
    "company",
    "year",
    "indicator_id",
    "pillar",
    "indicator_name",
    "definition",
    "framework",
    "retrieval_query",
    "evidence",
    "page_numbers",
    "evidence_length",
    "chunk_count",
    "page_count",
    "evidence_quality",
    "retrieval_score",
    "rerank_score",
    "candidate_count",
    "score",
    "confidence",
    "confidence_label",
    "disclosure_type",
    "disclosure_present",
    "is_quantitative",
    "has_target_or_outcome",
    "matched_keywords",
    "matched_negative_keywords",
    "matched_units",
    "matched_score_patterns",
    "matched_hard_required",
    "matched_soft_positive",
    "matched_qualitative_patterns",
    "matched_quantitative_patterns",
    "structural_score",
    "embedding_score",
    "keyword_score",
    "evidence_quality_score",
    "domain",
    "subdomain",
    "reasoning",
    "dense_candidate_count",
    "keyword_candidate_count",
    "merged_candidate_count",
    "after_relaxed_filter_count",
    "after_strict_filter_count",
    "final_candidate_count",
]


def load_codebook(indicator_dir: Path) -> pd.DataFrame:
    config = CodebookConfig()
    codebook_path = indicator_dir / config.filename
    if not codebook_path.exists():
        raise FileNotFoundError(f"ESG codebook not found: {codebook_path}")

    codebook = pd.read_csv(codebook_path)
    missing = [column for column in config.required_columns if column not in codebook.columns]
    if missing:
        raise ValueError(f"ESG codebook is missing required columns: {', '.join(missing)}")
    normalized = codebook.loc[:, list(config.required_columns)].copy()
    if "definition" in codebook.columns:
        normalized["definition"] = codebook["definition"].fillna("")
    else:
        normalized["definition"] = normalized.apply(_fallback_definition, axis=1)
    return normalized


def export_evidence_dataset(
    store: VectorStore,
    codebook: pd.DataFrame,
    firm_years: list[tuple[str, int | None]],
    output_path: Path | None = None,
    top_k: int = RetrievalConfig().top_k,
    reranker: BGEReranker | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reranker = reranker or BGEReranker()

    for company, year in firm_years:
        for _, indicator in codebook.iterrows():
            retrieval = store.retrieve_for_indicator(
                indicator,
                top_k=top_k,
                company=company,
                year=year,
                reranker=reranker,
            )
            bundle = build_evidence_bundle(str(indicator.indicator_id), retrieval["retrieved_chunks"])
            chunk_count = len(bundle["chunk_ids"])
            page_count = len(bundle["pages"])
            evidence = bundle["evidence_bundle"]
            rows.append(
                {
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
                    "page_count": page_count,
                    "evidence_quality": "LOW_EVIDENCE" if chunk_count < 2 else "OK",
                }
            )

    dataset = pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)
    path = output_path or PipelinePaths().evidence_dataset
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    return dataset


def _fallback_definition(indicator: pd.Series) -> str:
    parts = [
        str(indicator.get("indicator_name_vi", "")),
        str(indicator.get("retrieval_query", "")),
        str(indicator.get("keywords_vi", "")),
        str(indicator.get("evidence_requirement", "")),
    ]
    return " ".join(part for part in parts if part and part != "nan")
