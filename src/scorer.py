from __future__ import annotations


class ScoringDisabledError(RuntimeError):
    pass


def score_indicator(*_args, **_kwargs):
    raise ScoringDisabledError(
        "Scoring is disabled for the ChatGPT Plus workflow. "
        "Run the pipeline to export evidence_dataset.csv and chatgpt_batches/*.csv, "
        "then score manually inside ChatGPT Plus."
    )
