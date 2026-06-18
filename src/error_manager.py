from __future__ import annotations

import traceback as traceback_lib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class PipelineError:
    report: str
    stage: str
    error_type: str
    message: str
    traceback: str


class ErrorRecorder:
    def __init__(self) -> None:
        self.errors: list[PipelineError] = []

    def record(self, file: str | Path, stage: str, exc: BaseException) -> None:
        self.errors.append(
            PipelineError(
                report=str(file),
                stage=stage,
                error_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback_lib.format_exc(),
            )
        )

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(error) for error in self.errors], columns=["report", "stage", "error_type", "message", "traceback"]).to_csv(
            path, index=False
        )
