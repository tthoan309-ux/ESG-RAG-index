from __future__ import annotations

import numpy as np
import pandas as pd


def cohen_kappa(rater_a: list[int], rater_b: list[int]) -> float:
    if len(rater_a) != len(rater_b):
        raise ValueError("Rater lists must have the same length")
    labels = sorted(set(rater_a) | set(rater_b))
    if not labels:
        return float("nan")

    matrix = pd.crosstab(pd.Series(rater_a), pd.Series(rater_b), dropna=False).reindex(index=labels, columns=labels, fill_value=0)
    n = matrix.to_numpy().sum()
    observed = np.trace(matrix.to_numpy()) / n
    row_marginals = matrix.sum(axis=1).to_numpy() / n
    col_marginals = matrix.sum(axis=0).to_numpy() / n
    expected = float(np.dot(row_marginals, col_marginals))
    if expected == 1:
        return 1.0
    return float((observed - expected) / (1 - expected))


def external_correlation(esg_scores: pd.DataFrame, external_scores: pd.DataFrame, on: list[str] | None = None) -> float:
    keys = on or ["company", "year"]
    merged = esg_scores.merge(external_scores, on=keys, suffixes=("_rag", "_external"))
    if len(merged) < 2:
        return float("nan")
    return float(merged["ESG_rag"].corr(merged["ESG_external"]))

