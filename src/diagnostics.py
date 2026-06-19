from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_diagnostics(evidence: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if evidence.empty:
        return

    _write(score_distribution_by_indicator(evidence), output_dir / "score_distribution_by_indicator.csv")
    _write(confidence_by_indicator(evidence), output_dir / "confidence_by_indicator.csv")
    _write(score_distribution_by_pillar(evidence), output_dir / "score_distribution_by_pillar.csv")
    _write(distribution(evidence, "retrieval_score"), output_dir / "retrieval_score_distribution.csv")
    _write(distribution(evidence, "rerank_score"), output_dir / "rerank_score_distribution.csv")
    _write(distribution(evidence, "evidence_quality_score"), output_dir / "evidence_quality_distribution.csv")
    _write(candidate_funnel(evidence), output_dir / "candidate_funnel.csv")
    _write(false_negative_hotspots(evidence), output_dir / "false_negative_hotspots.csv")
    _write(ontology_coverage(evidence), output_dir / "ontology_coverage.csv")
    _write(review_sample(evidence), output_dir / "review_sample.csv")


def score_distribution_by_indicator(evidence: pd.DataFrame) -> pd.DataFrame:
    table = (
        evidence.pivot_table(
            index=["pillar", "indicator_id", "indicator_name"],
            columns="score",
            values="company",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for score in range(4):
        if score not in table.columns:
            table[score] = 0
    table["total"] = table[[0, 1, 2, 3]].sum(axis=1)
    table["positive_rate"] = ((table[[1, 2, 3]].sum(axis=1) / table["total"]).fillna(0)).round(4)
    return table


def confidence_by_indicator(evidence: pd.DataFrame) -> pd.DataFrame:
    grouped = evidence.groupby(["pillar", "indicator_id", "indicator_name", "confidence_label"], dropna=False).size()
    table = grouped.unstack(fill_value=0).reset_index()
    for label in ("LOW_CONFIDENCE", "MEDIUM_CONFIDENCE", "HIGH_CONFIDENCE"):
        if label not in table.columns:
            table[label] = 0
    averages = (
        evidence.groupby(["pillar", "indicator_id", "indicator_name"], dropna=False)["confidence"]
        .mean()
        .round(4)
        .rename("avg_confidence")
        .reset_index()
    )
    return table.merge(averages, on=["pillar", "indicator_id", "indicator_name"], how="left")


def score_distribution_by_pillar(evidence: pd.DataFrame) -> pd.DataFrame:
    return (
        evidence.groupby(["pillar", "score"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["pillar", "score"])
    )


def distribution(evidence: pd.DataFrame, column: str) -> pd.DataFrame:
    values = pd.to_numeric(evidence.get(column, pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return pd.DataFrame(
        [
            {
                "metric": column,
                "count": int(values.size),
                "mean": round(float(values.mean()), 4),
                "p10": round(float(values.quantile(0.10)), 4),
                "p25": round(float(values.quantile(0.25)), 4),
                "p50": round(float(values.quantile(0.50)), 4),
                "p75": round(float(values.quantile(0.75)), 4),
                "p90": round(float(values.quantile(0.90)), 4),
                "max": round(float(values.max()), 4),
            }
        ]
    )


def candidate_funnel(evidence: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dense_candidate_count",
        "keyword_candidate_count",
        "merged_candidate_count",
        "after_relaxed_filter_count",
        "after_strict_filter_count",
        "final_candidate_count",
    ]
    available = [column for column in columns if column in evidence.columns]
    if not available:
        return pd.DataFrame()
    return (
        evidence.groupby(["pillar", "indicator_id", "indicator_name"], dropna=False)[available]
        .mean()
        .round(3)
        .reset_index()
    )


def false_negative_hotspots(evidence: pd.DataFrame) -> pd.DataFrame:
    zero = evidence.loc[evidence["score"] == 0].copy()
    if zero.empty:
        return pd.DataFrame()
    for column in ("merged_candidate_count", "after_relaxed_filter_count", "after_strict_filter_count"):
        if column not in zero.columns:
            zero[column] = 0
    summary = (
        zero.groupby(["pillar", "indicator_id", "indicator_name"], dropna=False)
        .agg(
            score0_count=("score", "size"),
            avg_merged_candidates=("merged_candidate_count", "mean"),
            avg_relaxed_candidates=("after_relaxed_filter_count", "mean"),
            avg_strict_candidates=("after_strict_filter_count", "mean"),
            avg_retrieval_score=("retrieval_score", "mean"),
            avg_keyword_score=("keyword_score", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["score0_count", "avg_relaxed_candidates"], ascending=[False, True]).round(3)


def ontology_coverage(evidence: pd.DataFrame) -> pd.DataFrame:
    match_columns = [
        "matched_hard_required",
        "matched_soft_positive",
        "matched_qualitative_patterns",
        "matched_quantitative_patterns",
        "matched_units",
        "matched_negative_keywords",
    ]
    rows: list[dict] = []
    for keys, group in evidence.groupby(["pillar", "indicator_id", "indicator_name"], dropna=False):
        row = {"pillar": keys[0], "indicator_id": keys[1], "indicator_name": keys[2], "rows": len(group)}
        for column in match_columns:
            if column not in group.columns:
                row[f"{column}_hit_rate"] = 0.0
                continue
            row[f"{column}_hit_rate"] = round(float(group[column].fillna("").astype(str).str.len().gt(0).mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def review_sample(evidence: pd.DataFrame, per_bucket: int = 25) -> pd.DataFrame:
    sample_parts: list[pd.DataFrame] = []
    sort_cols = ["score", "confidence_label", "evidence_quality_score"]
    for _, group in evidence.sort_values(sort_cols, ascending=[True, True, False]).groupby(["score", "confidence_label"], dropna=False):
        sample_parts.append(group.head(per_bucket))
    if not sample_parts:
        return pd.DataFrame()
    columns = [
        "company",
        "year",
        "indicator_id",
        "indicator_name",
        "score",
        "confidence_label",
        "evidence_quality_score",
        "retrieval_score",
        "matched_keywords",
        "matched_hard_required",
        "matched_soft_positive",
        "matched_qualitative_patterns",
        "matched_quantitative_patterns",
        "evidence",
        "reasoning",
    ]
    available = [column for column in columns if column in evidence.columns]
    return pd.concat(sample_parts, ignore_index=True).loc[:, available]


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
