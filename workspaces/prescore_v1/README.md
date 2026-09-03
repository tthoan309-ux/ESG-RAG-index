# ESG pre-score workspace v1

This is a clean-room workspace beside the legacy ESG implementation. It reuses the annual-report corpus contract from `financialdistress`, retrieves auditable evidence, and then **stops before scoring**. No heuristic or model-generated ESG score is produced here.

## Boundary

| Owner | Responsibility |
|---|---|
| `financialdistress` | Download, extract/OCR, select report, quality audit, export source corpus |
| This workspace | Validate corpus, page-aware chunking, indicator retrieval, evidence/review packages |
| ChatGPT Plus + researcher | Apply the locked rubric, return the completed scoring CSV |

`RETRIEVAL_UNRESOLVED`, `CORPUS_UNREADABLE`, and `CORPUS_EMPTY` are review states. They are never converted to score 0 by this code.

## 1. Export the corpus

From `financialdistress`:

```bash
python scripts/export_esg_source_corpus.py \
  --input data/processed/annual_report_text_firm_year_2011_2025.parquet \
  --output data/processed/esg_source_corpus_2011_2025.parquet
```

Use the pre-cleaning firm-year text when possible so page break markers survive. If only flattened NLP text exists, the pipeline labels evidence `REPORT_LEVEL_ONLY` and leaves pages blank.

## 2. Build the pre-score package

From the `ESG-RAG-index` repository root:

```bash
python -m pip install -r workspaces/prescore_v1/requirements.txt
python workspaces/prescore_v1/run_prescore.py \
  --corpus /path/to/esg_source_corpus_2011_2025.parquet \
  --output-dir workspaces/prescore_v1/outputs/run_001
```

Default retrieval is deterministic BM25 and works on CPU. For semantic hybrid retrieval:

```bash
python -m pip install -r workspaces/prescore_v1/requirements-hybrid.txt
```

Then set `retrieval.mode: hybrid` in a copied config. The dense model is `BAAI/bge-m3`; record any config change as a new run.

## 3. Score with ChatGPT Plus

For each ZIP under `chatgpt_plus_batches/`, start a fresh chat and upload that one ZIP. Ask ChatGPT to follow `PROMPT.md`. Save the returned CSV against the batch's `SCORING_OUTPUT_TEMPLATE.csv`.

Do not edit `SCORING_INPUT.csv`, `CODEBOOK.yaml`, or `EVIDENCE.md` after packaging. `batch_manifest.csv` contains the ZIP hash for audit.

## 4. Optional API scoring pilot

The default workspace still stops before scoring. API scoring is a separate pilot path for controlled experiments after a pre-score run exists.

Build request payloads without calling the API:

```bash
python workspaces/prescore_v1/run_api_scoring.py \
  --provider openrouter \
  --model z-ai/glm-5.2:free \
  --run-dir workspaces/prescore_v1/outputs/pilot_2015_from_financialdistress \
  --output-dir workspaces/prescore_v1/outputs/api_scoring_openrouter_dry_run \
  --limit 1 \
  --dry-run
```

Run one real OpenRouter scoring call:

```bash
set OPENROUTER_API_KEY=sk-or-v1-...
python workspaces/prescore_v1/run_api_scoring.py \
  --provider openrouter \
  --model z-ai/glm-5.2:free \
  --run-dir workspaces/prescore_v1/outputs/pilot_2015_from_financialdistress \
  --output-dir workspaces/prescore_v1/outputs/api_scoring_openrouter_001 \
  --limit 1
```

OpenAI is still available with `--provider openai --model gpt-4o-mini` and `OPENAI_API_KEY`. The API path writes `api_scoring_rows.csv`, `api_validation_errors.csv`, `raw_api_outputs.jsonl`, and `api_scoring_manifest.json`. It validates score anchors, confidence labels, disclosure status, cited chunk IDs, and report-level page provenance. Keep this as an experiment until the final codebook and validation protocol are locked.

## Outputs

- `chunks.parquet`: deterministic page-aware chunks
- `evidence_candidates.parquet`: ranked evidence with retrieval diagnostics
- `scoring_rows.csv`: rubric/input rows; deliberately has no output score column
- `review_queue.csv`: unresolved corpus/retrieval cases
- `chatgpt_plus_batches/*.zip`: self-contained manual-scoring packages
- `batch_manifest.csv`: batch counts and SHA-256 hashes
- `run_manifest.json`: input/config/codebook hashes and `scoring_executed: false`

## Tests

```bash
pytest -q workspaces/prescore_v1/tests
```
