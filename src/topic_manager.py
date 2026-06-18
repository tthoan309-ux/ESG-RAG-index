from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ROOT
from .retriever import tokenize


@dataclass(frozen=True)
class Topic:
    topic_id: str
    name: str
    pillar: str
    indicators: tuple[str, ...]
    query: str


class TopicManager:
    def __init__(self, path: Path | None = None):
        self.path = path or (ROOT / "config" / "indicator_topics.yaml")
        self.topics = self._load_topics(self.path)

    def assign(self, codebook: pd.DataFrame) -> pd.DataFrame:
        frame = codebook.copy()
        topic_by_indicator = {
            indicator_id: topic.topic_id
            for topic in self.topics.values()
            for indicator_id in topic.indicators
        }
        frame["topic_id"] = frame["indicator_id"].map(lambda item: topic_by_indicator.get(str(item), self._auto_topic(item, frame)))
        frame["topic_name"] = frame["topic_id"].map(lambda item: self.topics[item].name if item in self.topics else str(item))
        frame["topic_query"] = frame["topic_id"].map(lambda item: self.topics[item].query if item in self.topics else "")
        return frame

    def topic_groups(self, codebook: pd.DataFrame) -> dict[str, pd.DataFrame]:
        assigned = self.assign(codebook)
        return {topic_id: group.copy() for topic_id, group in assigned.groupby("topic_id")}

    def _auto_topic(self, indicator_id: Any, frame: pd.DataFrame) -> str:
        row = frame.loc[frame["indicator_id"] == indicator_id]
        if row.empty:
            return "unmapped"
        query = " ".join(str(row.iloc[0].get(col, "")) for col in ("retrieval_query", "keywords_vi", "definition"))
        query_terms = set(tokenize(query))
        best_topic = None
        best_overlap = 0.0
        for topic in self.topics.values():
            topic_terms = set(tokenize(topic.query))
            overlap = len(query_terms & topic_terms) / max(len(query_terms | topic_terms), 1)
            if overlap > best_overlap:
                best_topic = topic.topic_id
                best_overlap = overlap
        return best_topic if best_topic and best_overlap >= 0.8 else f"auto_{str(indicator_id).lower()}"

    def _load_topics(self, path: Path) -> dict[str, Topic]:
        if not path.exists():
            return {}
        topics: dict[str, dict] = {}
        current: str | None = None
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.rstrip()
            if not line.strip() or line.strip() == "topics:":
                continue
            if line.startswith("  ") and not line.startswith("    "):
                current = line.strip().rstrip(":")
                topics[current] = {}
                continue
            if current and line.startswith("    ") and ":" in line:
                key, value = line.strip().split(":", 1)
                topics[current][key] = self._parse_value(value.strip())
        return {
            topic_id: Topic(
                topic_id=topic_id,
                name=str(values.get("name", topic_id)),
                pillar=str(values.get("pillar", "")),
                indicators=tuple(values.get("indicators", [])),
                query=str(values.get("query", "")),
            )
            for topic_id, values in topics.items()
        }

    def _parse_value(self, value: str):
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            return [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]
        return value.strip().strip("\"'")
