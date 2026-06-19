from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import pandas as pd

from .config import PipelinePaths, RetrievalConfig
from .evidence_builder import build_evidence_bundle
from .export_evidence import EVIDENCE_COLUMNS
from .manifest_manager import ManifestManager, ReportIdentity
from .progress import progress
from .reranker import BGEReranker
from .retriever import VectorStore, strip_accents, tokenize
from .storage import read_table, write_table
from .ontology import IndicatorOntology, OntologyManager
from .diagnostics import write_diagnostics


@dataclass
class RetrievalResult:
    evidence: pd.DataFrame
    retrieval_count: int
    reused_count: int


@dataclass(frozen=True)
class CandidateEvaluation:
    keep: bool
    strict_keep: bool
    relevance_score: float
    keyword_score: float
    structural_score: float
    embedding_score: float
    disclosure_present: bool
    disclosure_type: str
    is_quantitative: bool
    has_target_or_outcome: bool
    matched_keywords: tuple[str, ...]
    matched_negative_keywords: tuple[str, ...]
    matched_units: tuple[str, ...]
    matched_score_patterns: tuple[str, ...]
    matched_hard_required: tuple[str, ...]
    matched_soft_positive: tuple[str, ...]
    matched_qualitative_patterns: tuple[str, ...]
    matched_quantitative_patterns: tuple[str, ...]
    explanation: str


class RetrievalManager:
    def __init__(
        self,
        store: VectorStore,
        reranker: BGEReranker,
        config: RetrievalConfig,
        manifest: ManifestManager,
        cache_dir: Path | None = None,
        warehouse_dir: Path | None = None,
        runtime_path: Path | None = None,
        benchmark_path: Path | None = None,
        rebuild: bool = False,
        workers: int = 1,
    ):
        self.store = store
        self.reranker = reranker
        self.config = config
        self.manifest = manifest
        self.cache_dir = cache_dir or PipelinePaths().evidence_cache
        self.warehouse_dir = warehouse_dir or PipelinePaths().evidence_warehouse
        self.runtime_path = runtime_path or PipelinePaths().retrieval_runtime
        self.benchmark_path = benchmark_path or PipelinePaths().warehouse_benchmark
        self.rebuild = rebuild
        self.workers = workers
        self.config_hash = retrieval_config_hash(config)
        self.ontology_manager = OntologyManager()
        self._manifest_lock = Lock()

    def export_evidence(
        self,
        codebook: pd.DataFrame,
        firm_years: list[tuple[str, int | None]],
        output_path: Path | None = None,
    ) -> RetrievalResult:
        codebook = self.ontology_manager.attach(codebook)
        rows: list[dict] = []
        runtime_rows: list[dict] = []
        benchmark_rows: list[dict] = []
        reused = 0

        if self.workers <= 1:
            iterator = (self._retrieve_firm_year(company, year, codebook) for company, year in firm_years)
            for group_rows, group_runtime, group_benchmark, group_reused in progress(
                iterator, total=len(firm_years), desc="Retrieving", unit="firm-year"
            ):
                rows.extend(group_rows)
                runtime_rows.extend(group_runtime)
                benchmark_rows.append(group_benchmark)
                reused += group_reused
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = [executor.submit(self._retrieve_firm_year, company, year, codebook) for company, year in firm_years]
                for future in progress(as_completed(futures), total=len(futures), desc="Retrieving", unit="firm-year"):
                    group_rows, group_runtime, group_benchmark, group_reused = future.result()
                    rows.extend(group_rows)
                    runtime_rows.extend(group_runtime)
                    benchmark_rows.append(group_benchmark)
                    reused += group_reused

        evidence = pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)
        evidence = self._calibrate_confidence(evidence)
        write_table(output_path or PipelinePaths().evidence_dataset, evidence)
        write_table(self.runtime_path, pd.DataFrame(runtime_rows))
        write_table(self.benchmark_path, pd.DataFrame(benchmark_rows))
        write_diagnostics(evidence, PipelinePaths().root / "outputs" / "pipeline_artifacts" / "diagnostics")
        self.manifest.mark_exported_for_all_current()
        self.manifest.save()
        return RetrievalResult(evidence, len(rows), reused)

    def _retrieve_firm_year(
        self, company: str, year: int | None, codebook: pd.DataFrame
    ) -> tuple[list[dict], list[dict], dict, int]:
        report_id = f"{company}_{year}"
        report_hash = self._report_hash(report_id)
        rows: list[dict] = []
        runtimes: list[dict] = []
        reused = 0
        retrieval_calls_after = 0
        runtime_before = 0.0
        runtime_after = 0.0
        for _, indicator in codebook.iterrows():
            start = perf_counter()
            cache_path = self._indicator_cache_path(report_hash, str(indicator.indicator_id))
            if not self.rebuild and cache_path.exists():
                row = self._read_indicator_cache(cache_path)
                if row is not None:
                    rows.append(row)
                    reused += 1
                    runtimes.append(
                        {
                            "report": report_id,
                            "indicator": indicator.indicator_id,
                            "domain": row.get("domain", ""),
                            "subdomain": row.get("subdomain", ""),
                            "runtime_seconds": round(perf_counter() - start, 6),
                            "retrieval_score": row.get("retrieval_score", 0.0),
                            "rerank_score": row.get("rerank_score", 0.0),
                            "reranker_used": False,
                            "candidate_count": row.get("candidate_count", 0),
                            "confidence": row.get("confidence", 0.0),
                            "cache_reused": True,
                        }
                    )
                    continue
            chunks, search_runtime, funnel = self._retrieve_indicator_chunks(company, year, indicator)
            runtime_after += search_runtime
            retrieval_calls_after += 1
            row = self._row_from_chunks(company=company, year=year, indicator=indicator, chunks=chunks, funnel=funnel)
            self._write_indicator_cache(cache_path, row, report_hash)
            rows.append(row)
            runtimes.append(
                {
                    "report": report_id,
                    "indicator": indicator.indicator_id,
                    "domain": row.get("domain", ""),
                    "subdomain": row.get("subdomain", ""),
                    "runtime_seconds": round(perf_counter() - start, 6),
                    "retrieval_score": row.get("retrieval_score", 0.0),
                    "rerank_score": row.get("rerank_score", 0.0),
                    "reranker_used": False,
                    "candidate_count": row.get("candidate_count", 0),
                    "confidence": row.get("confidence", 0.0),
                    "cache_reused": False,
                }
            )

        before = len(codebook)
        benchmark = {
            "report": report_id,
            "retrieval_calls_before": before,
            "retrieval_calls_after": retrieval_calls_after,
            "runtime_before": round(runtime_before, 6),
            "runtime_after": round(runtime_after, 6),
            "ontology_count": len(codebook),
            "indicator_count": len(codebook),
        }
        self._mark_retrieved(report_id)
        return rows, runtimes, benchmark, reused

    def _retrieve_indicator_chunks(self, company: str, year: int | None, indicator: Any) -> tuple[list[dict], float, dict[str, int]]:
        start = perf_counter()
        ontology: IndicatorOntology = indicator.ontology
        query = self._indicator_query(indicator, ontology)
        candidate_idx = self.store._candidate_indexes(company=company, year=year)
        empty_funnel = {
            "dense_candidate_count": 0,
            "keyword_candidate_count": 0,
            "merged_candidate_count": 0,
            "after_relaxed_filter_count": 0,
            "after_strict_filter_count": 0,
            "final_candidate_count": 0,
        }
        if not candidate_idx:
            return [], perf_counter() - start, empty_funnel

        dense = self.store._embedding_search(query, candidate_idx, top_k=max(self.config.prefetch_k, 30))
        keyword_query = " ".join(ontology.expanded_query_terms)
        keyword = self.store._bm25.search(keyword_query, candidate_idx, top_k=self.config.warehouse_top_k)
        candidates = self._merge_candidates(dense, keyword)
        funnel = dict(empty_funnel)
        funnel["dense_candidate_count"] = len(dense)
        funnel["keyword_candidate_count"] = len(keyword)
        funnel["merged_candidate_count"] = len(candidates)

        evaluated: list[dict] = []
        strict_count = 0
        for candidate in candidates:
            item = dict(candidate)
            evaluation = self._evaluate_candidate(ontology, item)
            if not evaluation.keep:
                continue
            if evaluation.strict_keep:
                strict_count += 1
            item["indicator_query"] = query
            item["retrieval_score"] = float(evaluation.relevance_score)
            item["reranker_score"] = float(evaluation.relevance_score)
            item["reranker_model"] = "ontology-heuristic-reranker"
            item["keyword_score"] = float(evaluation.keyword_score)
            item["structural_score"] = float(evaluation.structural_score)
            item["normalized_embedding_score"] = float(evaluation.embedding_score)
            item["disclosure_present"] = evaluation.disclosure_present
            item["disclosure_type"] = evaluation.disclosure_type
            item["is_quantitative"] = evaluation.is_quantitative
            item["has_target_or_outcome"] = evaluation.has_target_or_outcome
            item["matched_keywords"] = "; ".join(evaluation.matched_keywords)
            item["matched_negative_keywords"] = "; ".join(evaluation.matched_negative_keywords)
            item["matched_units"] = "; ".join(evaluation.matched_units)
            item["matched_score_patterns"] = "; ".join(evaluation.matched_score_patterns)
            item["matched_hard_required"] = "; ".join(evaluation.matched_hard_required)
            item["matched_soft_positive"] = "; ".join(evaluation.matched_soft_positive)
            item["matched_qualitative_patterns"] = "; ".join(evaluation.matched_qualitative_patterns)
            item["matched_quantitative_patterns"] = "; ".join(evaluation.matched_quantitative_patterns)
            item["reranker_explanation"] = evaluation.explanation
            evaluated.append(item)

        chunks = self._dedupe_chunks(evaluated)[: self.config.top_k]
        for rank, chunk in enumerate(chunks, start=1):
            chunk["rank"] = rank
        funnel["after_relaxed_filter_count"] = len(evaluated)
        funnel["after_strict_filter_count"] = strict_count
        funnel["final_candidate_count"] = len(chunks)
        return chunks, perf_counter() - start, funnel

    def _merge_candidates(self, *groups: list[dict]) -> list[dict]:
        merged: dict[str, dict] = {}
        for group in groups:
            for item in group:
                key = str(item.get("chunk_id") or item.get("chunk_index"))
                existing = merged.setdefault(key, dict(item))
                existing.update({k: v for k, v in item.items() if k.endswith("_score") or k.endswith("_rank")})
        return list(merged.values())

    def _evaluate_candidate(self, ontology: IndicatorOntology, candidate: dict) -> CandidateEvaluation:
        text = str(candidate.get("text", ""))
        terms = self._content_terms(text)
        term_set = set(terms)
        normalized_text = " ".join(terms)
        hard_required = self._matched_phrases_in_terms(ontology.hard_required, terms, term_set, normalized_text, window=8, allow_single=True)
        soft_positive = self._matched_phrases_in_terms(ontology.soft_positive, terms, term_set, normalized_text, window=8, allow_single=True)
        qualitative = self._matched_phrases_in_terms(ontology.qualitative_patterns, terms, term_set, normalized_text, window=8, allow_single=False)
        quantitative = self._matched_phrases_in_terms(ontology.quantitative_patterns, terms, term_set, normalized_text, window=5, allow_single=True)
        positive = self._matched_phrases_in_terms(ontology.positive_keywords, terms, term_set, normalized_text, window=8, allow_single=True)
        alternatives = self._matched_phrases_in_terms(ontology.alternative_phrases, terms, term_set, normalized_text, window=8, allow_single=True)
        required = self._matched_phrases_in_terms(ontology.required_entities, terms, term_set, normalized_text, window=8, allow_single=True)
        negative = self._matched_phrases_in_terms(ontology.negative_keywords, terms, term_set, normalized_text, window=8, allow_single=False)
        false_positive = self._matched_phrases_in_terms(ontology.false_positive_patterns, terms, term_set, normalized_text, window=8, allow_single=False)
        units = self._matched_phrases_in_terms(ontology.units, terms, term_set, normalized_text, window=4, allow_single=True)
        target = self._matched_phrases_in_terms(
            ontology.score_patterns.get("target", ()) + ontology.target_patterns,
            terms,
            term_set,
            normalized_text,
            window=8,
            allow_single=True,
        )

        numeric = bool(re.search(r"(?<!\w)(\d+[\d.,]*\s*(%|tco2e|co2e|m3|m³|kwh|mwh|gj|mj|kg|ton|tấn|người|vnd|vnđ|đồng|usd|cases|vụ|giờ|hours)?)", strip_accents(text).lower()))
        keyword_hits = tuple(dict.fromkeys(hard_required + soft_positive + positive + alternatives + required + qualitative))
        pattern_hits = tuple(dict.fromkeys(qualitative + quantitative + target))
        has_indicator_signal = bool(hard_required or soft_positive or required or positive or alternatives)
        has_qualitative_signal = bool(has_indicator_signal and (qualitative or ontology.disclosure_type in {"policy", "governance", "certification"}))
        is_quantitative = bool(units or quantitative or (numeric and ontology.disclosure_type == "quantitative" and has_indicator_signal))
        has_target = bool(target) and self._target_applicable(ontology)

        embedding_score = max(float(candidate.get("embedding_score", 0.0) or 0.0), 0.0)
        embedding_score = min(embedding_score * 100.0, 100.0)
        keyword_score = min(
            len(hard_required) * 18
            + len(soft_positive) * 10
            + len(qualitative) * 8
            + len(quantitative) * 8
            + len(units) * 12
            + len(required) * 8
            + len(alternatives) * 5,
            100,
        )
        structural_score = self._structural_score(text, numeric=numeric, units=units)
        relevance = (
            0.42 * keyword_score
            + 0.25 * embedding_score
            + 0.18 * structural_score
            + 8 * int(has_qualitative_signal)
            + 9 * int(is_quantitative)
            + 5 * int(has_target)
        )

        if negative:
            relevance -= min(len(negative) * 8, 24)
        if false_positive and not hard_required and not units:
            relevance -= 30

        relaxed_keep = relevance >= 12 and bool(has_indicator_signal or units or quantitative or (embedding_score >= 45 and keyword_score > 0))
        strict_keep = bool(has_indicator_signal or units or quantitative) and relevance >= 28
        if negative and not hard_required and not units and relevance < 40:
            strict_keep = False
        if false_positive and not hard_required and not units:
            strict_keep = False
        disclosure_present = relaxed_keep and bool(has_indicator_signal or units or quantitative)
        if not disclosure_present:
            disclosure_type = "none"
        elif has_target and is_quantitative:
            disclosure_type = "target"
        elif is_quantitative:
            disclosure_type = "quantitative"
        else:
            disclosure_type = "qualitative"

        explanation_parts = []
        if keyword_hits:
            explanation_parts.append(f"matched indicator terms: {', '.join(keyword_hits[:6])}")
        if units:
            explanation_parts.append(f"matched units: {', '.join(units[:4])}")
        if target:
            explanation_parts.append(f"target/outcome signal: {', '.join(target[:4])}")
        if negative:
            explanation_parts.append(f"negative context: {', '.join(negative[:4])}")
        explanation = "; ".join(explanation_parts) or "no ontology signal"

        return CandidateEvaluation(
            keep=relaxed_keep,
            strict_keep=strict_keep,
            relevance_score=max(min(relevance, 100.0), 0.0),
            keyword_score=float(keyword_score),
            structural_score=float(structural_score),
            embedding_score=float(embedding_score),
            disclosure_present=disclosure_present,
            disclosure_type=disclosure_type,
            is_quantitative=is_quantitative,
            has_target_or_outcome=has_target,
            matched_keywords=tuple(keyword_hits),
            matched_negative_keywords=tuple(dict.fromkeys(negative + false_positive)),
            matched_units=tuple(dict.fromkeys(units)),
            matched_score_patterns=tuple(pattern_hits),
            matched_hard_required=tuple(dict.fromkeys(hard_required)),
            matched_soft_positive=tuple(dict.fromkeys(soft_positive)),
            matched_qualitative_patterns=tuple(dict.fromkeys(qualitative)),
            matched_quantitative_patterns=tuple(dict.fromkeys(quantitative)),
            explanation=explanation,
        )

    def _structural_score(self, text: str, numeric: bool, units: list[str]) -> float:
        score = 20.0
        if numeric:
            score += 30.0
        if units:
            score += 25.0
        if re.search(r"\b20\d{2}\b", text):
            score += 10.0
        if any(marker in text for marker in ("%", "|", ":", "\\t")):
            score += 10.0
        if len(text) >= 240:
            score += 5.0
        return min(score, 100.0)

    def _target_applicable(self, ontology: IndicatorOntology) -> bool:
        descriptor = " ".join(
            [
                ontology.indicator_id,
                ontology.name,
                ontology.domain,
                ontology.subdomain,
                ontology.disclosure_type,
                ontology.objective,
                ontology.formal_definition,
            ]
        )
        descriptor = strip_accents(descriptor).lower()
        target_terms = (
            "target",
            "goal",
            "reduction",
            "reduce",
            "outcome",
            "investment",
            "muc tieu",
            "giam",
            "ket qua",
            "dau tu",
        )
        return any(term in descriptor for term in target_terms)

    def _row_from_chunks(
        self,
        company: str,
        year: int | None,
        indicator: Any,
        chunks: list[dict],
        funnel: dict[str, int] | None = None,
    ) -> dict:
        ontology: IndicatorOntology = indicator.ontology
        bundle = build_evidence_bundle(str(indicator.indicator_id), chunks)
        evidence = bundle["evidence_bundle"]
        chunk_count = len(bundle["chunk_ids"])
        retrieval_score = max((float(chunk.get("retrieval_score", 0.0)) for chunk in chunks), default=0.0)
        rerank_score = max((float(chunk.get("reranker_score", 0.0)) for chunk in chunks), default=0.0)
        keyword_score = max((float(chunk.get("keyword_score", 0.0)) for chunk in chunks), default=0.0)
        embedding_score = max((float(chunk.get("normalized_embedding_score", 0.0)) for chunk in chunks), default=0.0)
        structural_score = max((float(chunk.get("structural_score", 0.0)) for chunk in chunks), default=0.0)
        evidence_quality_score = (0.4 * rerank_score) + (0.3 * embedding_score) + (0.2 * keyword_score) + (0.1 * structural_score)
        confidence_label = self._confidence_label(evidence_quality_score)
        score, disclosure_type, reasoning = self._score_indicator(chunks)
        matched_keywords = self._join_chunk_values(chunks, "matched_keywords")
        matched_negative = self._join_chunk_values(chunks, "matched_negative_keywords")
        matched_units = self._join_chunk_values(chunks, "matched_units")
        matched_patterns = self._join_chunk_values(chunks, "matched_score_patterns")
        matched_hard = self._join_chunk_values(chunks, "matched_hard_required")
        matched_soft = self._join_chunk_values(chunks, "matched_soft_positive")
        matched_qualitative = self._join_chunk_values(chunks, "matched_qualitative_patterns")
        matched_quantitative = self._join_chunk_values(chunks, "matched_quantitative_patterns")
        disclosure_present = bool(chunks and score > 0)
        is_quantitative = any(bool(chunk.get("is_quantitative")) for chunk in chunks)
        has_target = any(bool(chunk.get("has_target_or_outcome")) for chunk in chunks)
        return {
            "company": company,
            "year": year,
            "indicator_id": indicator.indicator_id,
            "pillar": indicator.pillar,
            "indicator_name": indicator.indicator_name_vi,
            "definition": indicator.definition,
            "framework": indicator.framework,
            "retrieval_query": indicator.retrieval_query,
            "evidence": evidence,
            "page_numbers": ",".join(bundle["pages"]),
            "evidence_length": len(evidence),
            "chunk_count": chunk_count,
            "page_count": len(bundle["pages"]),
            "evidence_quality": "LOW_EVIDENCE" if chunk_count < 2 else "OK",
            "retrieval_score": retrieval_score,
            "rerank_score": rerank_score,
            "candidate_count": len(chunks),
            "score": score,
            "confidence": round(evidence_quality_score / 100.0, 4),
            "confidence_label": confidence_label,
            "disclosure_type": disclosure_type,
            "disclosure_present": disclosure_present,
            "is_quantitative": is_quantitative,
            "has_target_or_outcome": has_target,
            "matched_keywords": matched_keywords,
            "matched_negative_keywords": matched_negative,
            "matched_units": matched_units,
            "matched_score_patterns": matched_patterns,
            "matched_hard_required": matched_hard,
            "matched_soft_positive": matched_soft,
            "matched_qualitative_patterns": matched_qualitative,
            "matched_quantitative_patterns": matched_quantitative,
            "structural_score": round(structural_score, 3),
            "embedding_score": round(embedding_score, 3),
            "keyword_score": round(keyword_score, 3),
            "evidence_quality_score": round(evidence_quality_score, 3),
            "domain": ontology.domain,
            "subdomain": ontology.subdomain,
            "reasoning": reasoning,
            **(funnel or {}),
        }

    def _indicator_query(self, indicator: Any, ontology: IndicatorOntology) -> str:
        parts = [
            str(getattr(indicator, "indicator_name_vi", "")),
            ontology.name,
            ontology.domain,
            ontology.subdomain,
            str(getattr(indicator, "definition", "")),
            str(getattr(indicator, "retrieval_query", "")),
            str(getattr(indicator, "keywords_vi", "")),
            str(getattr(indicator, "keywords_en", "")),
            " ".join(ontology.expanded_query_terms),
        ]
        return " ".join(part for part in parts if part and part != "nan")

    def _indicator_phrases(self, indicator: Any) -> list[str]:
        fields = [
            str(getattr(indicator, "indicator_name_vi", "")),
            str(getattr(indicator, "retrieval_query", "")),
            str(getattr(indicator, "keywords_vi", "")),
            str(getattr(indicator, "keywords_en", "")),
        ]
        phrases: list[str] = []
        for field in fields:
            for raw in field.replace("|", ";").split(";"):
                phrase = " ".join(self._content_terms(raw))
                if phrase and phrase not in phrases:
                    phrases.append(phrase)
        return phrases

    def _matched_phrases(self, phrases, text: str, window: int = 8, allow_single: bool = False) -> list[str]:
        terms = self._content_terms(text)
        term_set = set(terms)
        normalized_text = " ".join(terms)
        return self._matched_phrases_in_terms(phrases, terms, term_set, normalized_text, window=window, allow_single=allow_single)

    def _matched_phrases_in_terms(
        self,
        phrases,
        terms: list[str],
        term_set: set[str],
        normalized_text: str,
        window: int = 8,
        allow_single: bool = False,
    ) -> list[str]:
        matches: list[str] = []
        for raw_phrase in phrases:
            if self._looks_corrupt(raw_phrase):
                continue
            phrase = " ".join(self._content_terms(str(raw_phrase)))
            phrase_terms = phrase.split()
            if not phrase_terms:
                continue
            if len(phrase_terms) == 1:
                if allow_single and self._is_informative_single_term(phrase_terms[0]) and phrase_terms[0] in term_set:
                    matches.append(phrase)
                continue
            if phrase in normalized_text or self._terms_within_window(phrase_terms, terms, window):
                matches.append(phrase)
        return matches

    def _looks_corrupt(self, value) -> bool:
        text = str(value)
        return "?" in text or "�" in text or "Ä" in text or "Â" in text

    def _terms_within_window(self, phrase_terms: list[str], terms: list[str], window: int) -> bool:
        positions: list[list[int]] = []
        for phrase_term in phrase_terms:
            term_positions = [idx for idx, term in enumerate(terms) if term == phrase_term]
            if not term_positions:
                return False
            positions.append(term_positions)
        for first in positions[0]:
            if all(any(first <= pos <= first + window for pos in term_positions) for term_positions in positions[1:]):
                return True
        return False

    def _is_informative_single_term(self, term: str) -> bool:
        informative = {
            "ghg",
            "co2e",
            "tco2e",
            "ems",
            "ltifr",
            "ohs",
            "hse",
            "iso14001",
            "iso",
            "tcfd",
            "sbti",
            "csr",
            "esg",
            "vnd",
            "vnđ",
            "usd",
            "kwh",
            "mwh",
            "gj",
            "mj",
            "m3",
            "m³",
            "kg",
            "ton",
            "ha",
            "ml",
            "scope",
            "pham",
            "tu",
        }
        return term in informative or (any(char.isdigit() for char in term) and any(char.isalpha() for char in term)) or len(term) >= 6

    def _score_indicator(self, chunks: list[dict]) -> tuple[int, str, str]:
        if not chunks:
            return 0, "none", "No retained evidence after ontology filtering."
        has_disclosure = any(bool(chunk.get("disclosure_present")) for chunk in chunks)
        if not has_disclosure:
            return 0, "none", "Candidate chunks did not establish disclosure."

        has_quant = any(bool(chunk.get("is_quantitative")) for chunk in chunks)
        has_target = any(bool(chunk.get("has_target_or_outcome")) for chunk in chunks)
        if has_quant:
            if has_target:
                return 3, "target", "Evidence contains quantitative disclosure plus target/outcome signal."
            return 2, "quantitative", "Evidence contains numeric/unit-based disclosure for this indicator."
        return 1, "qualitative", "Evidence contains policy, commitment, governance mechanism, or narrative disclosure without sufficient numeric support."

    def _calibrate_confidence(self, evidence: pd.DataFrame) -> pd.DataFrame:
        if evidence.empty or "evidence_quality_score" not in evidence.columns:
            return evidence
        calibrated = evidence.copy()
        scores = pd.to_numeric(calibrated["evidence_quality_score"], errors="coerce").fillna(0.0)
        global_rank = scores.rank(method="average", pct=True)
        calibrated["confidence"] = 0.0
        calibrated["confidence_label"] = "LOW_CONFIDENCE"
        for indicator_id, group in calibrated.groupby("indicator_id", sort=False):
            group_scores = pd.to_numeric(group["evidence_quality_score"], errors="coerce").fillna(0.0)
            if group_scores.nunique() > 1:
                ranks = group_scores.rank(method="average", pct=True)
            else:
                ranks = global_rank.loc[group.index]
            for idx, percentile in ranks.items():
                if scores.loc[idx] <= 0 or int(calibrated.loc[idx, "score"]) == 0:
                    confidence = 0.0
                    label = "LOW_CONFIDENCE"
                else:
                    confidence = float(percentile)
                    if percentile >= 0.8:
                        label = "HIGH_CONFIDENCE"
                    elif percentile >= 0.4:
                        label = "MEDIUM_CONFIDENCE"
                    else:
                        label = "LOW_CONFIDENCE"
                calibrated.loc[idx, "confidence"] = round(confidence, 4)
                calibrated.loc[idx, "confidence_label"] = label
        return calibrated

    def _confidence_label(self, score: float) -> str:
        if score >= 70:
            return "HIGH_CONFIDENCE"
        if score >= 40:
            return "MEDIUM_CONFIDENCE"
        return "LOW_CONFIDENCE"

    def _join_chunk_values(self, chunks: list[dict], key: str) -> str:
        values: list[str] = []
        for chunk in chunks:
            for value in str(chunk.get(key, "")).split(";"):
                value = value.strip()
                if value and value not in values:
                    values.append(value)
        return "; ".join(values)

    def _content_terms(self, text: str) -> list[str]:
        stopwords = {
            "and",
            "or",
            "of",
            "the",
            "to",
            "in",
            "for",
            "on",
            "with",
            "by",
            "ve",
            "va",
            "cac",
            "cua",
            "cho",
            "trong",
            "ngoai",
            "cong",
            "bo",
            "muc",
            "tong",
            "don",
            "vi",
            "hoat",
            "dong",
            "chinh",
            "sach",
            "doanh",
            "thu",
            "hao",
            "quy",
            "tri",
            "chi",
            "sinh",
            "khu",
            "mua",
            "cam",
        }
        output: list[str] = []
        for term in tokenize(strip_accents(text)):
            if term in stopwords:
                continue
            if len(term) < 3 and not term.isdigit() and not self._is_informative_single_term(term):
                continue
            output.append(term)
        return output

    def _dedupe_chunks(self, chunks: list[dict]) -> list[dict]:
        seen: set[str] = set()
        output: list[dict] = []
        for chunk in sorted(chunks, key=lambda item: item.get("retrieval_score", 0.0), reverse=True):
            key = str(chunk.get("chunk_id") or chunk.get("text", "")[:160])
            if key in seen:
                continue
            seen.add(key)
            output.append(chunk)
        return output

    def _write_indicator_cache(self, path: Path, row: dict, report_hash: str) -> None:
        cached = dict(row)
        cached["report_hash"] = report_hash
        cached["retrieval_config_hash"] = self.config_hash
        write_table(path, pd.DataFrame([cached]))

    def _read_indicator_cache(self, path: Path) -> dict | None:
        try:
            frame = read_table(path)
        except Exception:
            return None
        if frame.empty:
            return None
        row = frame.iloc[0].to_dict()
        row.pop("report_hash", None)
        row.pop("retrieval_config_hash", None)
        return row

    def _indicator_cache_path(self, report_hash: str, indicator_id: str) -> Path:
        safe = indicator_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{report_hash}_{safe}_{self.config_hash}.parquet"

    def _report_hash(self, report_id: str) -> str:
        row = self.manifest.frame.loc[self.manifest.frame["report_id"] == report_id]
        if row.empty:
            return hashlib.sha256(report_id.encode("utf-8")).hexdigest()
        return str(row.iloc[0]["file_hash"])

    def _mark_retrieved(self, report_id: str) -> None:
        row = self.manifest.frame.loc[self.manifest.frame["report_id"] == report_id]
        if row.empty:
            return
        with self._manifest_lock:
            self.manifest.mark(ReportIdentity(report_id, str(row.iloc[0]["file_hash"]), Path("")), "retrieved")


def retrieval_config_hash(config: RetrievalConfig) -> str:
    payload = {
        "top_k": config.top_k,
        "prefetch_k": config.prefetch_k,
        "version": config.version,
        "reranker_model": config.reranker_model,
        "rerank_threshold": config.rerank_threshold,
        "retrieval_mode": config.retrieval_mode,
        "warehouse_top_k": config.warehouse_top_k,
        "indicator_specific_retrieval": True,
        "indicator_ontology": True,
        "multi_stage_dense_keyword": True,
        "heuristic_esg_reranker": True,
        "accent_normalized_tokens": True,
        "phrase_match_filter": True,
        "windowed_phrase_match": True,
        "ontology_schema": "hard-soft-qual-quant-v5-target-applicability",
        "qualitative_baseline_scoring": True,
        "candidate_funnel_logging": True,
        "indicator_percentile_confidence": True,
        "relaxed_pre_rerank_filter": True,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

