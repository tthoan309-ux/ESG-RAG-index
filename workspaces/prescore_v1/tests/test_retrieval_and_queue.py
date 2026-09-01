import pandas as pd

from esg_prescore.batching import build_scoring_rows
from esg_prescore.retrieval import retrieve_candidates


CODEBOOK = {
    "codebook_version": "test-v1",
    "indicators": [{
        "indicator_id": "C08", "name": "Target", "construct": "Target",
        "retrieval_queries": ["emissions target baseline year"],
        "evidence_requirements": ["target"], "exclusion_rules": ["generic"],
        "score_type": "ordinal", "rubric": {str(i): str(i) for i in range(5)},
    }],
}


def test_bm25_retrieves_relevant_chunk():
    chunks = pd.DataFrame([
        {"chunk_id": "a", "source_document_id": "d", "ticker": "AAA", "year": 2023,
         "page_number": 1, "provenance_status": "PAGE_LEVEL",
         "text": "Our emissions target uses 2020 as the baseline year."},
        {"chunk_id": "b", "source_document_id": "d", "ticker": "AAA", "year": 2023,
         "page_number": 2, "provenance_status": "PAGE_LEVEL", "text": "Revenue increased."},
    ])
    out = retrieve_candidates(chunks, CODEBOOK, top_k=2)
    assert out.iloc[0]["chunk_id"] == "a"
    assert "b" not in set(out["chunk_id"])


def test_no_candidate_is_review_not_zero():
    corpus = pd.DataFrame([{
        "ticker": "AAA", "year": 2023, "corpus_status": "READY",
    }])
    candidates = pd.DataFrame(columns=["ticker", "year", "indicator_id", "candidate_rank", "chunk_id"])
    rows, review = build_scoring_rows(corpus, candidates, CODEBOOK)
    assert rows.iloc[0]["pre_score_status"] == "RETRIEVAL_UNRESOLVED"
    assert len(review) == 1
    assert "score" not in rows.columns

