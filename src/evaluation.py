from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["company", "year", "indicator_id"]


def evaluate_scores(human_path: Path, ai_path: Path) -> pd.DataFrame:
    human = pd.read_csv(human_path)
    ai = pd.read_csv(ai_path)
    required = KEY_COLUMNS + ["score"]
    missing_human = [column for column in required if column not in human.columns]
    missing_ai = [column for column in required if column not in ai.columns]
    if missing_human or missing_ai:
        raise ValueError(f"Missing columns. human={missing_human}; ai={missing_ai}")

    merged = human[required].merge(ai[required], on=KEY_COLUMNS, suffixes=("_human", "_ai"))
    if merged.empty:
        raise ValueError("No overlapping firm-year-indicator rows found.")

    rows = []
    for label in sorted(set(merged["score_human"]) | set(merged["score_ai"])):
        tp = int(((merged["score_human"] == label) & (merged["score_ai"] == label)).sum())
        fp = int(((merged["score_human"] != label) & (merged["score_ai"] == label)).sum())
        fn = int(((merged["score_human"] == label) & (merged["score_ai"] != label)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({"score": label, "precision": precision, "recall": recall, "f1": f1, "support": int((merged["score_human"] == label).sum())})

    rows.append(
        {
            "score": "overall",
            "precision": float(np.mean([row["precision"] for row in rows])),
            "recall": float(np.mean([row["recall"] for row in rows])),
            "f1": float(np.mean([row["f1"] for row in rows])),
            "support": len(merged),
            "cohens_kappa": cohens_kappa(merged["score_human"], merged["score_ai"]),
            "krippendorff_alpha": krippendorff_alpha(merged[["score_human", "score_ai"]].to_numpy().T),
        }
    )
    return pd.DataFrame(rows)


def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    labels = sorted(set(a) | set(b))
    observed = float((a == b).mean())
    expected = 0.0
    for label in labels:
        expected += float((a == label).mean()) * float((b == label).mean())
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def krippendorff_alpha(ratings: np.ndarray) -> float:
    values = ratings[~pd.isna(ratings)]
    labels = sorted(set(values.tolist()))
    if len(labels) <= 1:
        return 1.0

    def distance(x, y) -> float:
        return 0.0 if x == y else 1.0

    disagreement = 0.0
    comparisons = 0
    for item_ratings in ratings.T:
        item_ratings = item_ratings[~pd.isna(item_ratings)]
        for i, left in enumerate(item_ratings):
            for right in item_ratings[i + 1 :]:
                disagreement += distance(left, right)
                comparisons += 1
    observed = disagreement / max(comparisons, 1)

    expected = 0.0
    total = len(values)
    for left in labels:
        for right in labels:
            expected += (np.sum(values == left) / total) * (np.sum(values == right) / total) * distance(left, right)
    return 1 - observed / expected if expected > 0 else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AI ESG scores against human-coded ESG scores.")
    parser.add_argument("human_scores", type=Path)
    parser.add_argument("ai_scores", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation_metrics.csv"))
    args = parser.parse_args()

    result = evaluate_scores(args.human_scores, args.ai_scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
