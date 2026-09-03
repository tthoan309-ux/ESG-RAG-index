from __future__ import annotations

import json

import pandas as pd

from esg_prescore.api_scoring import (
    APIScoringConfig,
    build_api_config,
    build_request_payload,
    normalize_scoring_result,
    run_api_scoring,
    validate_scoring_result,
)


def _row() -> dict:
    return {
        "scoring_row_id": "row_1",
        "ticker": "AAA",
        "year": 2023,
        "indicator_id": "C03",
        "indicator_name": "Scope 1 emissions",
        "construct": "Direct emissions",
        "score_type": "ordinal_disclosure_quality_0_4",
        "candidate_chunk_ids": "chunk_1",
        "codebook_version": "test-v1",
        "pre_score_status": "READY_FOR_CHATGPT",
    }


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "year": 2023,
                "indicator_id": "C03",
                "candidate_rank": 1,
                "chunk_id": "chunk_1",
                "page_number": 7,
                "provenance_status": "PAGE_LEVEL",
                "bm25_score": 1.2,
                "dense_score": pd.NA,
                "fusion_score": 1.2,
                "text": "Scope 1 emissions were 123 tCO2e in 2023.",
            }
        ]
    )


def _codebook() -> dict:
    return {
        "codebook_version": "test-v1",
        "indicators": [
            {
                "indicator_id": "C03",
                "name": "Scope 1 emissions",
                "construct": "Direct greenhouse-gas emissions disclosure",
                "retrieval_queries": ["Scope 1 emissions"],
                "evidence_requirements": ["Numeric Scope 1 value", "Unit and reporting period"],
                "exclusion_rules": ["Combined total where Scope 1 cannot be separated"],
                "score_type": "ordinal_disclosure_quality_0_4",
                "rubric": {str(index): f"anchor {index}" for index in range(5)},
            }
        ],
    }


def test_build_request_payload_uses_structured_json_schema():
    payload = build_request_payload(_row(), _candidates(), _codebook(), APIScoringConfig())
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["schema"]["properties"]["score"]["anyOf"][0]["enum"] == [0, 1, 2, 3, 4]
    user_content = json.loads(payload["input"][1]["content"])
    assert user_content["evidence"][0]["chunk_id"] == "chunk_1"
    assert "untrusted" in payload["input"][0]["content"]


def test_build_request_payload_supports_openrouter_chat_schema():
    config = APIScoringConfig(
        provider="openrouter",
        model="z-ai/glm-5.2:free",
        api_base="https://openrouter.ai/api/v1",
    )
    payload = build_request_payload(_row(), _candidates(), _codebook(), config)
    assert payload["model"] == "z-ai/glm-5.2:free"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"]["properties"]["score"]["anyOf"][0]["enum"] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert "messages" in payload
    assert "input" not in payload


def test_build_api_config_defaults_to_free_openrouter_model():
    config = build_api_config("openrouter")
    assert config.provider == "openrouter"
    assert config.model == "z-ai/glm-5.2:free"
    assert config.api_base == "https://openrouter.ai/api/v1"


def test_validate_rejects_unknown_evidence_chunk_id():
    output = normalize_scoring_result(
        _row(),
        {
            "score": 3,
            "confidence": "high",
            "disclosure_status": "DISCLOSED",
            "evidence_chunk_ids": "made_up_chunk",
            "evidence_pages": "7",
            "reasoning": "Numeric Scope 1 disclosure is present.",
        },
        APIScoringConfig(),
    )
    error = validate_scoring_result(output, _row(), _candidates())
    assert "unknown evidence_chunk_ids" in error


def test_api_scoring_dry_run_writes_no_scored_rows(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "api"
    batch_dir = run_dir / "chatgpt_plus_batches" / "batch_0001"
    batch_dir.mkdir(parents=True)
    pd.DataFrame([_row()]).to_csv(run_dir / "scoring_rows.csv", index=False)
    _candidates().to_parquet(run_dir / "evidence_candidates.parquet", index=False)
    (run_dir / "run_manifest.json").write_text(json.dumps({"scoring_executed": False}), encoding="utf-8")
    import yaml

    (batch_dir / "CODEBOOK.yaml").write_text(yaml.safe_dump(_codebook()), encoding="utf-8")

    manifest = run_api_scoring(run_dir, output_dir, dry_run=True, limit=1)
    assert manifest["dry_run"] is True
    assert manifest["scoring_executed"] is False
    assert (output_dir / "raw_api_outputs.jsonl").exists()
    assert pd.read_csv(output_dir / "api_scoring_rows.csv").empty
