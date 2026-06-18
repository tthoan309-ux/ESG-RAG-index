from __future__ import annotations


def build_explanation(company: str, year: int | None, score_result: dict, evidence_bundle: dict) -> dict:
    return {
        "company": company,
        "year": year,
        "indicator_id": score_result.get("indicator_id", evidence_bundle.get("indicator_id", "")),
        "score": score_result.get("score", 0),
        "evidence": score_result.get("evidence", ""),
        "supporting_quotes": extract_supporting_quotes(score_result.get("evidence", ""), evidence_bundle.get("evidence_bundle", "")),
        "page_numbers": ",".join(evidence_bundle.get("pages", [])),
    }


def indicator_evidence_rows(company: str, year: int | None, indicator_id: str, evidence_bundle: dict) -> list[dict]:
    rows: list[dict] = []
    pages = evidence_bundle.get("pages", [])
    evidence = evidence_bundle.get("evidence_bundle", "")
    for page in pages or [""]:
        rows.append(
            {
                "company": company,
                "year": year,
                "indicator_id": indicator_id,
                "page_number": page,
                "evidence": evidence,
            }
        )
    return rows


def extract_supporting_quotes(answer_evidence: str, bundle: str, max_quotes: int = 3) -> str:
    if answer_evidence.strip():
        return answer_evidence.strip()

    quotes = []
    for block in bundle.split("\n\n---\n\n"):
        text = block.split("Evidence:", 1)[-1].strip()
        if text:
            quotes.append(text[:500])
        if len(quotes) >= max_quotes:
            break
    return "\n---\n".join(quotes)
