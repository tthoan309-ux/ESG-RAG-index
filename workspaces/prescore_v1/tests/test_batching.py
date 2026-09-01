import zipfile

import pandas as pd

from esg_prescore.batching import build_scoring_rows, write_chatgpt_batches


def test_batch_contains_locked_files_but_input_has_no_score(tmp_path):
    codebook = {
        "codebook_version": "test-v1",
        "indicators": [{"indicator_id": "C01", "name": "Governance"}],
    }
    rows = pd.DataFrame([{
        "scoring_row_id": "row1", "ticker": "AAA", "year": 2023,
        "indicator_id": "C01", "indicator_name": "Governance", "construct": "Oversight",
        "score_type": "ordinal", "rubric_json": "{}", "evidence_requirements_json": "[]",
        "exclusion_rules_json": "[]", "candidate_chunk_ids": "chunk1", "evidence_count": 1,
        "pre_score_status": "READY_FOR_CHATGPT", "pre_score_reason": "ready",
        "codebook_version": "test-v1",
    }])
    candidates = pd.DataFrame([{
        "ticker": "AAA", "year": 2023, "indicator_id": "C01", "candidate_rank": 1,
        "chunk_id": "chunk1", "page_number": 5, "provenance_status": "PAGE_LEVEL",
        "text": "The board reviews climate risks quarterly.",
    }])
    manifest = write_chatgpt_batches(tmp_path, rows, candidates, codebook, "locked prompt")
    archive = tmp_path / manifest.iloc[0]["zip_path"]
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        assert names == {"SCORING_INPUT.csv", "EVIDENCE.md", "CODEBOOK.yaml", "PROMPT.md", "SCORING_OUTPUT_TEMPLATE.csv"}
        header = handle.read("SCORING_INPUT.csv").decode("utf-8").splitlines()[0].split(",")
        assert "score" not in header


def test_batch_keeps_all_indicators_for_one_firm_year_together(tmp_path):
    codebook = {
        "codebook_version": "test-v1",
        "indicators": [
            {
                "indicator_id": f"C{index:02d}",
                "name": f"Indicator {index}",
                "construct": "Carbon disclosure",
                "retrieval_queries": ["climate"],
                "evidence_requirements": ["evidence"],
                "exclusion_rules": [],
                "score_type": "ordinal",
                "rubric": {str(anchor): str(anchor) for anchor in range(5)},
            }
            for index in range(1, 13)
        ],
    }
    corpus = pd.DataFrame([{"ticker": "AAA", "year": 2023, "corpus_status": "READY"}])
    candidates = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "year": 2023,
                "indicator_id": f"C{index:02d}",
                "candidate_rank": 1,
                "chunk_id": f"chunk-{index}",
                "page_number": 1,
                "provenance_status": "PAGE_LEVEL",
                "text": "climate evidence",
            }
            for index in range(1, 13)
        ]
    )
    rows, _ = build_scoring_rows(corpus, candidates, codebook)
    manifest = write_chatgpt_batches(tmp_path, rows, candidates, codebook, "locked prompt")
    assert len(manifest) == 1
    with zipfile.ZipFile(tmp_path / manifest.iloc[0]["zip_path"]) as handle:
        input_rows = pd.read_csv(handle.open("SCORING_INPUT.csv"))
    assert input_rows["indicator_id"].tolist() == [f"C{index:02d}" for index in range(1, 13)]
