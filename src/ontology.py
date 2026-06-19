from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelinePaths


@dataclass(frozen=True)
class IndicatorOntology:
    indicator_id: str
    name: str
    pillar: str
    domain: str
    subdomain: str
    frameworks: tuple[str, ...] = field(default_factory=tuple)
    disclosure_type: str = "policy"
    formal_definition: str = ""
    objective: str = ""
    hard_required: tuple[str, ...] = field(default_factory=tuple)
    soft_positive: tuple[str, ...] = field(default_factory=tuple)
    qualitative_patterns: tuple[str, ...] = field(default_factory=tuple)
    quantitative_patterns: tuple[str, ...] = field(default_factory=tuple)
    required_entities: tuple[str, ...] = field(default_factory=tuple)
    positive_keywords: tuple[str, ...] = field(default_factory=tuple)
    negative_keywords: tuple[str, ...] = field(default_factory=tuple)
    alternative_phrases: tuple[str, ...] = field(default_factory=tuple)
    units: tuple[str, ...] = field(default_factory=tuple)
    score_patterns: dict[str, tuple[str, ...]] = field(default_factory=dict)
    target_patterns: tuple[str, ...] = field(default_factory=tuple)
    required_evidence_patterns: tuple[str, ...] = field(default_factory=tuple)
    false_positive_patterns: tuple[str, ...] = field(default_factory=tuple)
    validation_rules: dict[str, tuple[str, ...]] = field(default_factory=dict)
    score_0: str = "no disclosure"
    score_1: str = "qualitative disclosure"
    score_2: str = "quantitative disclosure"
    score_3: str = "quantitative disclosure with target or outcome"

    @property
    def expanded_query_terms(self) -> tuple[str, ...]:
        terms: list[str] = []
        for group in (
            self.hard_required,
            self.soft_positive,
            self.quantitative_patterns,
            self.required_entities,
            self.positive_keywords,
            self.alternative_phrases,
            self.units,
            self.score_patterns.get("qualitative", ()),
            self.score_patterns.get("quantitative", ()),
            self.score_patterns.get("target", ()),
        ):
            for item in group:
                if item and item not in terms:
                    terms.append(item)
        return tuple(terms)


class OntologyManager:
    def __init__(self, ontology_dir: Path | None = None):
        self.ontology_dir = ontology_dir or PipelinePaths().indicator_ontology
        self._cache: dict[str, IndicatorOntology] = {}

    def attach(self, codebook: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for _, row in codebook.iterrows():
            ontology = self.load(str(row["indicator_id"]), row)
            record = row.to_dict()
            record["ontology"] = ontology
            record["domain"] = ontology.domain
            record["subdomain"] = ontology.subdomain
            record["disclosure_type"] = ontology.disclosure_type
            record["formal_definition"] = ontology.formal_definition
            rows.append(record)
        return pd.DataFrame(rows)

    def load(self, indicator_id: str, codebook_row: pd.Series | None = None) -> IndicatorOntology:
        if indicator_id in self._cache:
            return self._cache[indicator_id]

        path = self.ontology_dir / f"{indicator_id}.yaml"
        if not path.exists():
            if codebook_row is None:
                raise FileNotFoundError(f"Indicator ontology not found: {path}")
            ontology = self._fallback(indicator_id, codebook_row)
        else:
            payload = _read_yaml(path)
            required_entities = _tuple(payload.get("required_entities"))
            positive_keywords = _tuple(payload.get("positive_keywords"))
            alternative_phrases = _tuple(payload.get("alternative_phrases"))
            if codebook_row is not None:
                codebook_terms = tuple(
                    _split_terms(str(codebook_row.get(column, "")))
                    for column in ("retrieval_query", "keywords_vi", "keywords_en", "indicator_name_vi", "definition")
                )
                flattened_codebook_terms = tuple(term for group in codebook_terms for term in group)
                positive_keywords = _unique(positive_keywords + flattened_codebook_terms)
                alternative_phrases = _unique(alternative_phrases + flattened_codebook_terms)
            units = _tuple(payload.get("units"))
            score_patterns = {key: _tuple(value) for key, value in dict(payload.get("score_patterns", {})).items()}
            disclosure_type = str(payload.get("disclosure_type", "policy"))
            qualitative_patterns = _specific_qualitative_patterns(
                _tuple(payload.get("qualitative_patterns")) or score_patterns.get("qualitative", ())
            )
            ontology = IndicatorOntology(
                indicator_id=str(payload.get("indicator_id", payload.get("indicator", indicator_id))),
                name=str(payload.get("name", indicator_id)),
                pillar=str(payload.get("pillar", "")),
                domain=str(payload.get("domain", "")),
                subdomain=str(payload.get("subdomain", "")),
                frameworks=_tuple(payload.get("frameworks")),
                disclosure_type=disclosure_type,
                formal_definition=str(payload.get("formal_definition", payload.get("definition", ""))),
                objective=str(payload.get("objective", "")),
                hard_required=_tuple(payload.get("hard_required")) or required_entities,
                soft_positive=_unique(_tuple(payload.get("soft_positive")) + positive_keywords + alternative_phrases),
                qualitative_patterns=_unique(qualitative_patterns + _qualitative_defaults(disclosure_type)),
                quantitative_patterns=_quantitative_patterns_for_type(
                    disclosure_type,
                    _tuple(payload.get("quantitative_patterns")),
                    units,
                    score_patterns.get("quantitative", ()),
                    qualitative_patterns + positive_keywords + alternative_phrases,
                ),
                required_entities=required_entities,
                positive_keywords=positive_keywords,
                negative_keywords=_tuple(payload.get("negative_keywords")),
                alternative_phrases=alternative_phrases,
                units=units,
                score_patterns=score_patterns,
                target_patterns=_tuple(payload.get("target_patterns")),
                required_evidence_patterns=_tuple(payload.get("required_evidence_patterns")),
                false_positive_patterns=_tuple(payload.get("false_positive_patterns")),
                validation_rules={key: _tuple(value) for key, value in dict(payload.get("validation_rules", {})).items()},
                score_0=str(payload.get("score_0", "no disclosure")),
                score_1=str(payload.get("score_1", "qualitative disclosure")),
                score_2=str(payload.get("score_2", "quantitative disclosure")),
                score_3=str(payload.get("score_3", "quantitative disclosure with target or outcome")),
            )

        self._cache[indicator_id] = ontology
        return ontology

    def _fallback(self, indicator_id: str, row: pd.Series) -> IndicatorOntology:
        keywords = _split_terms(str(row.get("keywords_vi", ""))) + _split_terms(str(row.get("keywords_en", "")))
        query_terms = _split_terms(str(row.get("retrieval_query", "")))
        return IndicatorOntology(
            indicator_id=indicator_id,
            name=str(row.get("indicator_name_vi", indicator_id)),
            pillar=str(row.get("pillar", "")),
            domain=str(row.get("category_vi", "")),
            subdomain="",
            formal_definition=str(row.get("definition", "")),
            objective=str(row.get("definition", "")),
            positive_keywords=tuple(dict.fromkeys(keywords + query_terms)),
            alternative_phrases=tuple(query_terms),
            required_entities=tuple(keywords[:3]),
            hard_required=tuple(keywords[:3]),
            soft_positive=tuple(dict.fromkeys(keywords + query_terms)),
            qualitative_patterns=_qualitative_defaults("policy"),
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"Could not read ontology YAML {path}: {exc}") from exc


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value if str(item).strip())


def _split_terms(value: str) -> list[str]:
    terms: list[str] = []
    for part in value.replace("|", ";").split(";"):
        part = part.strip()
        if part:
            terms.append(part)
    return terms


def _unique(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    output: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in output:
            output.append(item)
    return tuple(output)


def _qualitative_defaults(disclosure_type: str) -> tuple[str, ...]:
    common = ("code of conduct", "risk management system", "management system")
    if disclosure_type in {"policy", "governance", "certification"}:
        return common
    return ()


def _specific_quantitative_patterns(values: tuple[str, ...]) -> tuple[str, ...]:
    generic = {"%", "amount", "total", "ratio", "rate", "number", "volume", "value"}
    output: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item.lower() in generic:
            continue
        if item not in output:
            output.append(item)
    return tuple(output)


def _quantitative_patterns_for_type(
    disclosure_type: str,
    configured: tuple[str, ...],
    units: tuple[str, ...],
    score_patterns: tuple[str, ...],
    qualitative_terms: tuple[str, ...],
) -> tuple[str, ...]:
    if disclosure_type in {"policy", "governance", "certification"}:
        return _specific_quantitative_patterns(units)
    qualitative = {item.lower() for item in qualitative_terms}
    values = _specific_quantitative_patterns(configured or (units + score_patterns))
    return tuple(item for item in values if item.lower() not in qualitative or item in units)


def _specific_qualitative_patterns(values: tuple[str, ...]) -> tuple[str, ...]:
    generic = {
        "policy",
        "commitment",
        "management",
        "program",
        "procedure",
        "governance",
        "committee",
        "oversight",
        "strategy",
        "training",
        "risk management",
        "system",
        "mechanism",
        "initiative",
        "practice",
        "disclosure",
        "chinh sach",
        "cam ket",
        "quan ly",
        "chuong trinh",
        "quy trinh",
        "uy ban",
        "giam sat",
        "dao tao",
    }
    output: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item.lower() in generic:
            continue
        if item not in output:
            output.append(item)
    return tuple(output)
