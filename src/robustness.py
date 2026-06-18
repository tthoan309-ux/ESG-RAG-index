from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PipelinePaths


def build_robustness_report(
    indicator_scores: pd.DataFrame,
    output_path: Path | None = None,
    top_k_values: tuple[int, ...] = (3, 5, 10),
    prompt_versions: tuple[str, ...] = ("v1.0", "v2.0"),
) -> pd.DataFrame:
    rows: list[dict] = []
    base_score = float(indicator_scores["score"].mean()) if not indicator_scores.empty else 0.0
    for top_k in top_k_values:
        for prompt_version in prompt_versions:
            rows.append(
                {
                    "top_k": top_k,
                    "prompt_version": prompt_version,
                    "mean_score_reference": base_score,
                    "status": "requires_explicit_rescore",
                }
            )

    report = pd.DataFrame(rows)
    path = output_path or PipelinePaths().robustness_report
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    return report
