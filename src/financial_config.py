from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PipelinePaths


@dataclass(frozen=True)
class FinancialIndicator:
    indicator_id: str
    label_vi: str
    statement_group: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    retrieval_queries: tuple[str, ...] = field(default_factory=tuple)
    unit_candidates: tuple[str, ...] = field(default_factory=tuple)
    normalization_rules: dict[str, Any] = field(default_factory=dict)

    @property
    def query_text(self) -> str:
        parts = [self.label_vi, *self.aliases, *self.retrieval_queries]
        return " ".join(part for part in parts if part)


class FinancialIndicatorConfig:
    def __init__(self, path: Path | None = None):
        self.path = path or PipelinePaths().financial_indicators
        self._cache: dict[str, FinancialIndicator] | None = None

    def load(self) -> dict[str, FinancialIndicator]:
        if self._cache is not None:
            return self._cache
        payload = _read_yaml(self.path)
        indicators: dict[str, FinancialIndicator] = {}
        for indicator_id, item in payload.items():
            indicators[str(indicator_id)] = FinancialIndicator(
                indicator_id=str(indicator_id),
                label_vi=str(item.get("label_vi", indicator_id)),
                statement_group=str(item.get("statement_group", "general")),
                aliases=_tuple(item.get("aliases")),
                retrieval_queries=_tuple(item.get("retrieval_queries")),
                unit_candidates=_tuple(item.get("unit_candidates")),
                normalization_rules=dict(item.get("normalization_rules", {})),
            )
        self._cache = indicators
        return indicators

    def by_group(self) -> dict[str, list[FinancialIndicator]]:
        groups: dict[str, list[FinancialIndicator]] = {}
        for indicator in self.load().values():
            groups.setdefault(indicator.statement_group, []).append(indicator)
        return groups


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"Could not read financial indicator config {path}: {exc}") from exc


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value if str(item).strip())
