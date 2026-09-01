from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_INDICATOR_FIELDS = {
    "indicator_id", "name", "construct", "retrieval_queries",
    "evidence_requirements", "exclusion_rules", "score_type", "rubric",
}
CARBON_V1_INDICATORS = {f"C{index:02d}" for index in range(1, 13)}


def load_codebook(path: str | Path) -> dict[str, Any]:
    codebook = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(codebook, dict) or not isinstance(codebook.get("indicators"), list):
        raise ValueError("Codebook must contain an indicators list.")
    if "disclosure quality" not in str(codebook.get("method_note", "")).lower():
        raise ValueError("Codebook must state that it measures disclosure quality.")
    if "performance" not in str(codebook.get("method_note", "")).lower():
        raise ValueError("Codebook must distinguish disclosure quality from environmental performance.")
    if "compliance" not in str(codebook.get("method_note", "")).lower():
        raise ValueError("Codebook must state that it is not compliance certification.")
    if "adequate report coverage" not in str(codebook.get("zero_rule", "")).lower():
        raise ValueError("Codebook must define the adequate-coverage condition for score 0.")
    if "RETRIEVAL_UNRESOLVED" not in str(codebook.get("zero_rule", "")):
        raise ValueError("Codebook must keep missing retrieval evidence unresolved.")
    identifiers: list[str] = []
    for position, indicator in enumerate(codebook["indicators"]):
        missing = sorted(REQUIRED_INDICATOR_FIELDS - set(indicator))
        if missing:
            raise ValueError(f"Indicator {position} missing: {', '.join(missing)}")
        identifiers.append(str(indicator["indicator_id"]))
        rubric = indicator["rubric"]
        if set(map(str, rubric)) != {"0", "1", "2", "3", "4"}:
            raise ValueError(f"{indicator['indicator_id']} rubric must define anchors 0..4.")
        if not indicator["retrieval_queries"]:
            raise ValueError(f"{indicator['indicator_id']} requires retrieval queries.")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("indicator_id values must be unique.")
    if set(identifiers) != CARBON_V1_INDICATORS or len(identifiers) != 12:
        raise ValueError("Carbon codebook v1 must contain exactly indicators C01..C12.")
    return codebook


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
