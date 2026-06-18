from __future__ import annotations


def build_evidence_bundle(indicator_id: str, retrieved_chunks: list[dict]) -> dict:
    seen: set[str] = set()
    evidence_parts: list[str] = []
    pages: list[str] = []
    chunk_ids: list[str] = []
    sources: list[str] = []

    for chunk in retrieved_chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        text = str(chunk.get("text", "")).strip()
        source = str(chunk.get("source_file", ""))
        page = str(chunk.get("page", ""))
        dedupe_key = chunk_id or f"{source}:{page}:{text[:120]}"

        if not text or dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        chunk_ids.append(chunk_id)
        if page and page not in pages:
            pages.append(page)
        if source and source not in sources:
            sources.append(source)

        evidence_parts.append(
            "\n".join(
                [
                    f"Source: {source}",
                    f"Page: {page}",
                    f"Chunk ID: {chunk_id}",
                    f"Reranker score: {chunk.get('reranker_score', '')}",
                    "Evidence:",
                    text,
                ]
            )
        )

    return {
        "indicator_id": indicator_id,
        "company": retrieved_chunks[0].get("company", "") if retrieved_chunks else "",
        "year": retrieved_chunks[0].get("year") if retrieved_chunks else None,
        "evidence_bundle": "\n\n---\n\n".join(evidence_parts),
        "pages": pages,
        "chunk_ids": chunk_ids,
        "source_documents": sources,
    }
