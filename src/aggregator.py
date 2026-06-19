from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import PipelinePaths


INDICATOR_COLUMNS = [
    "company",
    "year",
    "indicator_id",
    "pillar",
    "indicator_name",
    "score",
    "confidence",
    "confidence_label",
    "disclosure_type",
    "evidence_quality_score",
    "retrieval_score",
    "rerank_score",
    "matched_keywords",
    "page_numbers",
    "reasoning",
]


def build_index_from_evidence(
    evidence_path: Path | None = None,
    index_output_path: Path | None = None,
    indicator_output_path: Path | None = None,
    pillar_output_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = PipelinePaths()
    evidence_path = evidence_path or paths.evidence_dataset
    evidence = pd.read_csv(evidence_path)
    indicator_scores = prepare_indicator_scores(evidence)

    index_output_path = index_output_path or (paths.root / "outputs" / "esg_index.csv")
    indicator_output_path = indicator_output_path or (paths.root / "outputs" / "indicator_score_panel.csv")
    pillar_output_path = pillar_output_path or (paths.root / "outputs" / "pipeline_artifacts" / "ESG_scores" / "pillar_breakdown.csv")

    index, pillar_breakdown = aggregate_index(indicator_scores)
    _write(indicator_scores, indicator_output_path)
    _write(pillar_breakdown, pillar_output_path)
    _write(index, index_output_path)
    return indicator_scores, pillar_breakdown, index


def prepare_indicator_scores(evidence: pd.DataFrame) -> pd.DataFrame:
    missing = {"company", "year", "indicator_id", "pillar", "score"} - set(evidence.columns)
    if missing:
        raise ValueError(f"Evidence dataset is missing required columns: {', '.join(sorted(missing))}")

    panel = evidence.copy()
    panel["score"] = pd.to_numeric(panel["score"], errors="raise").astype(int)
    invalid = panel.loc[~panel["score"].isin([0, 1, 2, 3]), "score"].drop_duplicates().tolist()
    if invalid:
        raise ValueError(f"Invalid ESG score values: {invalid}")

    panel["confidence"] = pd.to_numeric(panel.get("confidence", 0.0), errors="coerce").fillna(0.0)
    panel["evidence_quality_score"] = pd.to_numeric(panel.get("evidence_quality_score", 0.0), errors="coerce").fillna(0.0)
    panel = panel.sort_values(["company", "year", "indicator_id", "evidence_quality_score"], ascending=[True, True, True, False])
    panel = panel.drop_duplicates(["company", "year", "indicator_id"], keep="first")

    for column in INDICATOR_COLUMNS:
        if column not in panel.columns:
            panel[column] = ""
    return panel.loc[:, INDICATOR_COLUMNS].sort_values(["company", "year", "indicator_id"]).reset_index(drop=True)


def aggregate_index(indicator_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = indicator_scores.copy()
    panel["score"] = pd.to_numeric(panel["score"], errors="raise").astype(int)
    panel["confidence"] = pd.to_numeric(panel.get("confidence", 0.0), errors="coerce").fillna(0.0)
    panel["disclosed"] = panel["score"].gt(0).astype(int)
    panel["high_confidence"] = panel.get("confidence_label", "").eq("HIGH_CONFIDENCE").astype(int)
    panel["medium_confidence"] = panel.get("confidence_label", "").eq("MEDIUM_CONFIDENCE").astype(int)
    panel["low_confidence"] = panel.get("confidence_label", "").eq("LOW_CONFIDENCE").astype(int)

    pillar = (
        panel.groupby(["company", "year", "pillar"], dropna=False)
        .agg(
            raw_score=("score", "sum"),
            indicator_count=("indicator_id", "nunique"),
            disclosed_count=("disclosed", "sum"),
            high_confidence_count=("high_confidence", "sum"),
            medium_confidence_count=("medium_confidence", "sum"),
            low_confidence_count=("low_confidence", "sum"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    pillar["max_score"] = pillar["indicator_count"] * 3
    pillar["pillar_index_0_100"] = (pillar["raw_score"] / pillar["max_score"] * 100).round(4)
    pillar["disclosure_rate"] = (pillar["disclosed_count"] / pillar["indicator_count"]).round(4)
    pillar["avg_confidence"] = pillar["avg_confidence"].round(4)

    index = (
        panel.groupby(["company", "year"], dropna=False)
        .agg(
            total_raw_score=("score", "sum"),
            indicator_count=("indicator_id", "nunique"),
            disclosed_count=("disclosed", "sum"),
            high_confidence_count=("high_confidence", "sum"),
            medium_confidence_count=("medium_confidence", "sum"),
            low_confidence_count=("low_confidence", "sum"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    index["total_max_score"] = index["indicator_count"] * 3
    index["esg_index_equal_indicator"] = (index["total_raw_score"] / index["total_max_score"] * 100).round(4)
    index["disclosure_rate"] = (index["disclosed_count"] / index["indicator_count"]).round(4)
    index["avg_confidence"] = index["avg_confidence"].round(4)

    pillar_wide = pillar.pivot_table(index=["company", "year"], columns="pillar", values="pillar_index_0_100", fill_value=0).reset_index()
    for pillar_name in ("E", "S", "G"):
        if pillar_name not in pillar_wide.columns:
            pillar_wide[pillar_name] = 0.0
    pillar_wide = pillar_wide.rename(columns={"E": "environment_index", "S": "social_index", "G": "governance_index"})
    pillar_wide["esg_index_equal_pillar"] = pillar_wide[["environment_index", "social_index", "governance_index"]].mean(axis=1).round(4)

    index = index.merge(pillar_wide, on=["company", "year"], how="left")
    ordered = [
        "company",
        "year",
        "esg_index_equal_indicator",
        "esg_index_equal_pillar",
        "environment_index",
        "social_index",
        "governance_index",
        "total_raw_score",
        "total_max_score",
        "indicator_count",
        "disclosed_count",
        "disclosure_rate",
        "avg_confidence",
        "high_confidence_count",
        "medium_confidence_count",
        "low_confidence_count",
    ]
    return index.loc[:, ordered].sort_values(["company", "year"]).reset_index(drop=True), pillar.sort_values(["company", "year", "pillar"])


def aggregate_scores(indicator_scores: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    index, _ = aggregate_index(indicator_scores)
    legacy = index.rename(
        columns={
            "environment_index": "E",
            "social_index": "S",
            "governance_index": "G",
            "esg_index_equal_pillar": "ESG",
        }
    )
    legacy = legacy[["company", "year", "E", "S", "G", "ESG"]].sort_values(["company", "year"])
    _write(legacy, output_path)
    return legacy


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ESG Disclosure Index from evidence_dataset.csv.")
    parser.add_argument("--evidence", type=Path, default=PipelinePaths().evidence_dataset)
    parser.add_argument("--index-output", type=Path, default=PipelinePaths().root / "outputs" / "esg_index.csv")
    parser.add_argument("--indicator-output", type=Path, default=PipelinePaths().root / "outputs" / "indicator_score_panel.csv")
    parser.add_argument(
        "--pillar-output",
        type=Path,
        default=PipelinePaths().root / "outputs" / "pipeline_artifacts" / "ESG_scores" / "pillar_breakdown.csv",
    )
    args = parser.parse_args()
    indicator_scores, pillar_breakdown, index = build_index_from_evidence(
        evidence_path=args.evidence,
        index_output_path=args.index_output,
        indicator_output_path=args.indicator_output,
        pillar_output_path=args.pillar_output,
    )
    print(f"Wrote {len(index)} firm-year ESG index rows to {args.index_output}")
    print(f"Wrote {len(indicator_scores)} indicator score rows to {args.indicator_output}")
    print(f"Wrote {len(pillar_breakdown)} pillar rows to {args.pillar_output}")


if __name__ == "__main__":
    main()
