from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from esg_prescore.codebook import CARBON_V1_INDICATORS, load_codebook
from esg_prescore.pipeline import run_prescore
from esg_prescore.retrieval import retrieve_candidates


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _corpus_fixture() -> pd.DataFrame:
    ready_text = (
        "The board has climate governance oversight and reviews climate risks quarterly. "
        "---PAGE_BREAK--- "
        "Scope 1 emissions were 123 tCO2e in 2023. "
        "The company set an emissions target to reduce greenhouse gas emissions by 30 percent by 2030."
    )
    return pd.DataFrame(
        [
            {
                "source_document_id": "doc_aaa_2023",
                "source_segment_id": "seg_aaa_2023_p1",
                "ticker": "AAA",
                "year": 2023,
                "page_number": 1,
                "text": ready_text.split("---PAGE_BREAK---")[0],
                "corpus_status": "READY",
                "provenance_status": "PAGE_LEVEL",
            },
            {
                "source_document_id": "doc_aaa_2023",
                "source_segment_id": "seg_aaa_2023_p2",
                "ticker": "AAA",
                "year": 2023,
                "page_number": 2,
                "text": ready_text.split("---PAGE_BREAK---")[1],
                "corpus_status": "READY",
                "provenance_status": "PAGE_LEVEL",
            },
            {
                "source_document_id": "doc_bad_2023",
                "source_segment_id": "seg_bad_2023",
                "ticker": "BAD",
                "year": 2023,
                "page_number": pd.NA,
                "text": "",
                "corpus_status": "CORPUS_UNREADABLE",
                "provenance_status": "REPORT_LEVEL_ONLY",
            },
        ]
    )


def test_carbon_codebook_v1_has_exact_required_indicators():
    codebook = load_codebook(WORKSPACE_ROOT / "config" / "codebook_carbon_v1.yaml")
    assert {indicator["indicator_id"] for indicator in codebook["indicators"]} == CARBON_V1_INDICATORS
    assert len(codebook["indicators"]) == 12
    assert "disclosure quality" in codebook["method_note"].lower()
    assert "RETRIEVAL_UNRESOLVED" in codebook["zero_rule"]


def test_retrieval_does_not_cross_firm_year():
    codebook = {
        "codebook_version": "test-v1",
        "method_note": "disclosure quality not environmental performance",
        "zero_rule": "Score 0 requires adequate report coverage; otherwise RETRIEVAL_UNRESOLVED.",
        "indicators": [
            {
                "indicator_id": "C08",
                "name": "Target",
                "construct": "Target",
                "retrieval_queries": ["emissions target baseline year"],
                "evidence_requirements": ["target"],
                "exclusion_rules": ["generic"],
                "score_type": "ordinal",
                "rubric": {str(i): str(i) for i in range(5)},
            }
        ],
    }
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "aaa-target",
                "source_document_id": "doc1",
                "ticker": "AAA",
                "year": 2023,
                "page_number": 1,
                "provenance_status": "PAGE_LEVEL",
                "text": "AAA has an emissions target baseline year.",
            },
            {
                "chunk_id": "bbb-target",
                "source_document_id": "doc2",
                "ticker": "BBB",
                "year": 2024,
                "page_number": 1,
                "provenance_status": "PAGE_LEVEL",
                "text": "BBB has an emissions target baseline year.",
            },
        ]
    )
    out = retrieve_candidates(chunks, codebook, top_k=1)
    assert set(out["ticker"]) == {"AAA", "BBB"}
    assert set(zip(out["ticker"], out["year"], out["chunk_id"])) == {
        ("AAA", 2023, "aaa-target"),
        ("BBB", 2024, "bbb-target"),
    }


def test_pipeline_runs_to_chatgpt_zip_and_stops_before_scoring(tmp_path):
    corpus_path = tmp_path / "source_corpus.parquet"
    output_dir = tmp_path / "run_001"
    _corpus_fixture().to_parquet(corpus_path, index=False)

    manifest = run_prescore(
        corpus_path=corpus_path,
        output_dir=output_dir,
        config_path=WORKSPACE_ROOT / "config" / "prescore.yaml",
        workspace_root=WORKSPACE_ROOT,
    )

    assert (output_dir / "chunks.parquet").exists()
    assert (output_dir / "evidence_candidates.parquet").exists()
    assert (output_dir / "scoring_rows.csv").exists()
    assert (output_dir / "review_queue.csv").exists()
    assert (output_dir / "batch_manifest.csv").exists()
    assert (output_dir / "run_manifest.json").exists()
    assert manifest["scoring_executed"] is False
    assert json.loads((output_dir / "run_manifest.json").read_text())["scoring_executed"] is False

    scoring_rows = pd.read_csv(output_dir / "scoring_rows.csv")
    review_queue = pd.read_csv(output_dir / "review_queue.csv")
    forbidden = {"score", "confidence", "reasoning", "disclosure_status"}
    assert not (forbidden & set(scoring_rows.columns))
    assert review_queue["ticker"].eq("BAD").any()
    assert not any(path.name.lower().startswith("esg_score") for path in output_dir.iterdir())

    batch_manifest = pd.read_csv(output_dir / "batch_manifest.csv")
    zip_path = output_dir / batch_manifest.iloc[0]["zip_path"]
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        assert set(archive.namelist()) == {
            "SCORING_INPUT.csv",
            "EVIDENCE.md",
            "CODEBOOK.yaml",
            "PROMPT.md",
            "SCORING_OUTPUT_TEMPLATE.csv",
        }


def test_pipeline_refuses_to_overwrite_existing_output_dir(tmp_path):
    corpus_path = tmp_path / "source_corpus.parquet"
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    _corpus_fixture().to_parquet(corpus_path, index=False)

    with pytest.raises(FileExistsError):
        run_prescore(
            corpus_path=corpus_path,
            output_dir=output_dir,
            config_path=WORKSPACE_ROOT / "config" / "prescore.yaml",
            workspace_root=WORKSPACE_ROOT,
        )
