from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .storage import read_table, write_table


STAGES = ("parsed", "chunked", "embedded", "retrieved", "exported")


@dataclass(frozen=True)
class ReportIdentity:
    report_id: str
    file_hash: str
    path: Path


class ManifestManager:
    def __init__(self, path: Path):
        self.path = path
        self.frame = self._load()

    def _load(self) -> pd.DataFrame:
        columns = ["report_id", "file_hash", *STAGES, "timestamp"]
        if not self.path.exists():
            return pd.DataFrame(columns=columns)
        frame = read_table(self.path)
        for column in columns:
            if column not in frame.columns:
                frame[column] = False if column in STAGES else ""
        return frame[columns]

    def save(self) -> None:
        write_table(self.path, self.frame)

    def identity(self, report: Path) -> ReportIdentity:
        return ReportIdentity(report.stem, sha256_file(report), report)

    def is_complete(self, identity: ReportIdentity, stage: str) -> bool:
        row = self._row(identity.report_id)
        if row is None or str(row["file_hash"]) != identity.file_hash:
            return False
        return bool(row.get(stage, False))

    def changed_or_new(self, identity: ReportIdentity) -> bool:
        row = self._row(identity.report_id)
        return row is None or str(row["file_hash"]) != identity.file_hash

    def mark(self, identity: ReportIdentity, stage: str, value: bool = True) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown manifest stage: {stage}")
        idx = self._row_index(identity.report_id)
        now = datetime.now(timezone.utc).isoformat()
        if idx is None:
            row = {"report_id": identity.report_id, "file_hash": identity.file_hash, "timestamp": now}
            row.update({name: False for name in STAGES})
            self.frame = pd.concat([self.frame, pd.DataFrame([row])], ignore_index=True)
            idx = len(self.frame) - 1
        if str(self.frame.loc[idx, "file_hash"]) != identity.file_hash:
            self.frame.loc[idx, "file_hash"] = identity.file_hash
            for name in STAGES:
                self.frame.loc[idx, name] = False
        self.frame.loc[idx, stage] = value
        self.frame.loc[idx, "timestamp"] = now

    def mark_exported_for_all_current(self) -> None:
        if len(self.frame):
            self.frame.loc[:, "exported"] = True
            self.frame.loc[:, "timestamp"] = datetime.now(timezone.utc).isoformat()

    def _row(self, report_id: str) -> pd.Series | None:
        idx = self._row_index(report_id)
        if idx is None:
            return None
        return self.frame.loc[idx]

    def _row_index(self, report_id: str) -> int | None:
        matches = self.frame.index[self.frame["report_id"] == report_id].tolist()
        return int(matches[0]) if matches else None


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()
