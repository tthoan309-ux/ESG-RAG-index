from __future__ import annotations

import math
import shutil
import json
from pathlib import Path

import pandas as pd

from .config import PipelinePaths
from .prompt_generator import write_batch_prompt


DEFAULT_BATCH_SIZE = 100
MANIFEST_COLUMNS = ["batch_id", "file_name", "record_count", "start_indicator", "end_indicator"]


def build_chatgpt_batches(
    evidence_dataset_path: Path | None = None,
    output_dir: Path | None = None,
    json_output_dir: Path | None = None,
    prompt_package_dir: Path | None = None,
    manifest_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    paths = PipelinePaths()
    dataset_path = evidence_dataset_path or paths.evidence_dataset
    batches_dir = output_dir or paths.chatgpt_batches
    json_dir = json_output_dir or paths.chatgpt_batches_json
    package_dir = prompt_package_dir or paths.chatgpt_prompt_package
    manifest_output = manifest_path or paths.chatgpt_batch_manifest

    dataset = pd.read_csv(dataset_path)
    batches_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir(parents=True, exist_ok=True)
    _clear_old_batches(batches_dir)
    _clear_old_batches(json_dir)
    _clear_old_batches(package_dir)

    manifest_rows: list[dict] = []
    total_batches = math.ceil(len(dataset) / batch_size) if len(dataset) else 0
    for batch_index in range(total_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, len(dataset))
        batch = dataset.iloc[start:end].copy()
        batch_id = f"batch_{batch_index + 1:03d}"
        file_name = f"chatgpt_batch_{batch_index + 1:03d}.csv"
        json_file_name = f"chatgpt_batch_{batch_index + 1:03d}.json"
        prompt_file_name = f"chatgpt_batch_{batch_index + 1:03d}_prompt.md"
        batch_path = batches_dir / file_name
        json_path = json_dir / json_file_name
        package_csv_path = package_dir / file_name
        prompt_path = package_dir / prompt_file_name
        batch.to_csv(batch_path, index=False)
        batch.to_csv(package_csv_path, index=False)
        json_path.write_text(json.dumps(batch.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
        write_batch_prompt(batch, prompt_path, batch_id)

        manifest_rows.append(
            {
                "batch_id": batch_id,
                "file_name": str(batch_path),
                "record_count": len(batch),
                "start_indicator": batch.iloc[0]["indicator_id"] if len(batch) else "",
                "end_indicator": batch.iloc[-1]["indicator_id"] if len(batch) else "",
            }
        )

    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_output, index=False)
    return manifest


def _clear_old_batches(output_dir: Path) -> None:
    for path in output_dir.glob("chatgpt_batch_*.csv"):
        if path.is_file():
            path.unlink()
    for path in output_dir.glob("chatgpt_batch_*.json"):
        if path.is_file():
            path.unlink()
    for path in output_dir.glob("chatgpt_batch_*_prompt.md"):
        if path.is_file():
            path.unlink()
    metadata_dir = output_dir / "_metadata"
    if metadata_dir.exists():
        shutil.rmtree(metadata_dir)
