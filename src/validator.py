from __future__ import annotations


class ScoreValidationError(ValueError):
    pass


def validate_score_response(result: dict) -> dict:
    score = result.get("score")
    confidence = result.get("confidence")
    evidence = str(result.get("evidence", "")).strip()

    if score not in {0, 1, 2, 3}:
        raise ScoreValidationError(f"score must be one of 0, 1, 2, 3; got {score!r}")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise ScoreValidationError(f"confidence must be in [0, 1]; got {confidence!r}")
    if not evidence:
        raise ScoreValidationError("evidence must not be empty")

    result["score"] = int(score)
    result["confidence"] = float(confidence)
    result["evidence"] = evidence
    result["reasoning"] = str(result.get("reasoning", ""))
    return result


def review_row(company: str, year: int | None, indicator_id: str, reason: str, payload: dict | None = None) -> dict:
    return {
        "company": company,
        "year": year,
        "indicator_id": indicator_id,
        "review_reason": reason,
        "payload": payload or {},
    }
