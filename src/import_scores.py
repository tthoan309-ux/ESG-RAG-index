from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .aggregator import aggregate_scores
from .config import PipelinePaths
from .export_evidence import load_codebook


REQUIRED_SCORE_COLUMNS = ["company", "year", "indicator_id", "score", "confidence", "reasoning"]


def import_chatgpt_scores(
    scores_path: Path,
    output_scores_path: Path | None = None,
    output_esg_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = PipelinePaths()
    imported = _read_scores(scores_path)
    validated = _validate_scores(imported)
    codebook = load_codebook(paths.indicators)[["indicator_id", "pillar"]].drop_duplicates()

    indicator_scores = validated.merge(codebook, on="indicator_id", how="left")
    if indicator_scores["pillar"].isna().any():
        missing = indicator_scores.loc[indicator_scores["pillar"].isna(), "indicator_id"].drop_duplicates().tolist()
        raise ValueError(f"Imported scores contain indicator_id values missing from codebook: {missing}")

    indicator_scores = indicator_scores[["company", "year", "indicator_id", "pillar", "score", "confidence", "reasoning"]]
    scores_output = output_scores_path or (paths.indicator_scores / "indicator_scores.csv")
    scores_output.parent.mkdir(parents=True, exist_ok=True)
    indicator_scores.to_csv(scores_output, index=False)

    esg_output = output_esg_path or (paths.esg_scores / "esg_scores.csv")
    esg_scores = aggregate_scores(indicator_scores, esg_output)
    return indicator_scores, esg_scores


def _read_scores(path: Path) -> pd.DataFrame:
    if path.is_dir():
        frames = [pd.read_csv(file) for file in sorted(path.glob("*.csv"))]
        if not frames:
            raise FileNotFoundError(f"No CSV score files found in {path}")
        return pd.concat(frames, ignore_index=True)
    return pd.read_csv(path)


def _validate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_SCORE_COLUMNS if column not in scores.columns]
    if missing:
        raise ValueError(f"Score file is missing required columns: {', '.join(missing)}")

    cleaned = scores.loc[:, REQUIRED_SCORE_COLUMNS].copy()
    cleaned["score"] = pd.to_numeric(cleaned["score"], errors="raise").astype(int)
    invalid_scores = cleaned.loc[~cleaned["score"].isin([0, 1, 2, 3])]
    if not invalid_scores.empty:
        raise ValueError(f"Invalid score values found: {invalid_scores['score'].drop_duplicates().tolist()}")

    cleaned["confidence"] = pd.to_numeric(cleaned["confidence"], errors="raise").astype(float)
    invalid_confidence = cleaned.loc[(cleaned["confidence"] < 0) | (cleaned["confidence"] > 1)]
    if not invalid_confidence.empty:
        raise ValueError("confidence must be within [0, 1]")

    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Import ChatGPT Plus scoring CSV files and aggregate ESG index.")
    parser.add_argument("scores_path", type=Path, help="CSV file or directory of CSV files returned by ChatGPT Plus.")
    args = parser.parse_args()

    indicator_scores, esg_scores = import_chatgpt_scores(args.scores_path)
    print(f"Imported {len(indicator_scores)} indicator scores.")
    print(f"Wrote ESG scores for {len(esg_scores)} firm-year rows.")


if __name__ == "__main__":
    main()
