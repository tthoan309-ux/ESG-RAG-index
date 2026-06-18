from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import PipelinePaths
from .export_evidence import load_codebook
from .topic_manager import TopicManager


def build_warehouse_benchmark(root: Path | None = None) -> pd.DataFrame:
    paths = PipelinePaths(root=root or PipelinePaths().root)
    codebook = TopicManager().assign(load_codebook(paths.indicators))
    raw_reports = sorted(paths.raw_reports.glob("*.pdf"))
    topics = codebook["topic_id"].nunique()
    indicators = len(codebook)
    before = len(raw_reports) * indicators
    after = len(raw_reports) * topics
    warehouse_files = list(paths.evidence_warehouse.glob("*.parquet")) if paths.evidence_warehouse.exists() else []
    expected_warehouse = max(after, 1)
    warehouse_hit_rate = min(len(warehouse_files), expected_warehouse) / expected_warehouse * 100
    call_reduction = (1 - after / max(before, 1)) * 100

    previous_runtime = _previous_retrieval_runtime(paths, before)
    measured_after = _latest_retrieval_runtime(paths)
    estimated_after = measured_after if measured_after > 0 else previous_runtime * (after / max(before, 1))
    frame = pd.DataFrame(
        [
            {
                "retrieval_calls_before": before,
                "retrieval_calls_after": after,
                "warehouse_hit_rate": round(warehouse_hit_rate, 3),
                "runtime_before": round(previous_runtime, 3),
                "runtime_after": round(estimated_after, 3),
                "call_reduction_percent": round(call_reduction, 3),
                "reports": len(raw_reports),
                "indicators": indicators,
                "topics": topics,
                "warehouse_files": len(warehouse_files),
            }
        ]
    )
    paths.warehouse_benchmark.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.warehouse_benchmark, index=False)
    return frame


def _previous_retrieval_runtime(paths: PipelinePaths, expected_calls: int) -> float:
    legacy_files = [
        path
        for path in paths.evidence_cache.glob("*.parquet")
        if _is_legacy_indicator_cache(path)
    ]
    if len(legacy_files) > 1:
        times = [path.stat().st_mtime for path in legacy_files]
        avg = (max(times) - min(times)) / max(len(legacy_files), 1)
        if avg > 0:
            return avg * expected_calls
    runtime_profile = paths.root / "outputs" / "runtime_profile.csv"
    if runtime_profile.exists():
        frame = pd.read_csv(runtime_profile)
        row = frame.loc[frame["stage"] == "Retrieval"]
        if not row.empty:
            return float(row.iloc[0]["runtime_seconds"])
    return 0.0


def _is_legacy_indicator_cache(path: Path) -> bool:
    parts = path.stem.split("_")
    return len(parts) == 3 and parts[1].isdigit() and parts[2][:1] in {"E", "S", "G"}


def _latest_retrieval_runtime(paths: PipelinePaths) -> float:
    manifest = paths.root / "outputs" / "run_manifest.json"
    if not manifest.exists():
        return 0.0
    try:
        import json

        data = json.loads(manifest.read_text(encoding="utf-8"))
        return float(data.get("retrieval_time_seconds", 0.0) or 0.0)
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate topic warehouse retrieval call reduction.")
    parser.add_argument("--root", type=Path, default=PipelinePaths().root)
    args = parser.parse_args()
    print(build_warehouse_benchmark(args.root).to_string(index=False))


if __name__ == "__main__":
    main()
