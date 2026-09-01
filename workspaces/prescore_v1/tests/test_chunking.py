import pandas as pd

from esg_prescore.chunking import chunk_corpus


def test_chunk_ids_are_deterministic_and_page_is_preserved():
    corpus = pd.DataFrame([{
        "source_document_id": "doc1", "source_segment_id": "seg1",
        "ticker": "AAA", "year": 2023, "page_number": 7,
        "provenance_status": "PAGE_LEVEL", "corpus_status": "READY",
        "text": "word " * 130,
    }])
    first = chunk_corpus(corpus, target_words=100, overlap_words=20)
    second = chunk_corpus(corpus, target_words=100, overlap_words=20)
    assert first["chunk_id"].tolist() == second["chunk_id"].tolist()
    assert first["page_number"].tolist() == [7, 7]
    assert first.iloc[1]["word_start"] == 80

