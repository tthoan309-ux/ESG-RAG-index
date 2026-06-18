from __future__ import annotations

from pathlib import Path

import pandas as pd


def aggregate_scores(indicator_scores: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    max_by_pillar = indicator_scores.groupby("pillar")["indicator_id"].nunique() * 3
    grouped = indicator_scores.groupby(["company", "year", "pillar"], dropna=False)["score"].sum().reset_index()
    grouped["subindex"] = grouped.apply(lambda row: row["score"] / max_by_pillar[row["pillar"]], axis=1)

    pivot = grouped.pivot_table(index=["company", "year"], columns="pillar", values="subindex", fill_value=0).reset_index()
    for pillar in ("E", "S", "G"):
        if pillar not in pivot.columns:
            pivot[pillar] = 0.0
    pivot["ESG"] = pivot[["E", "S", "G"]].mean(axis=1)
    pivot = pivot[["company", "year", "E", "S", "G", "ESG"]].sort_values(["company", "year"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(output_path, index=False)
    return pivot

