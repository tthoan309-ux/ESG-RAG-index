from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class StageTimer:
    durations: dict[str, float] = field(default_factory=dict)
    _starts: dict[str, float] = field(default_factory=dict)

    def start(self, stage: str) -> None:
        self._starts[stage] = perf_counter()

    def stop(self, stage: str) -> float:
        elapsed = perf_counter() - self._starts.pop(stage, perf_counter())
        self.durations[stage] = self.durations.get(stage, 0.0) + elapsed
        return elapsed

    def get(self, stage: str) -> float:
        return self.durations.get(stage, 0.0)


def per_hour(count: int, seconds: float) -> float:
    return 0.0 if seconds <= 0 else count / seconds * 3600
