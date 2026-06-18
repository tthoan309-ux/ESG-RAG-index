# ESG-RAG-Index

A reproducible evidence-preparation pipeline for constructing a firm-year ESG Disclosure Quality Index from annual, sustainability, ESG, or integrated reports.

This version is designed for a ChatGPT Plus workflow. It does not call the OpenAI API and does not score indicators automatically inside the pipeline.

```text
Annual Report
  -> Parser
  -> Chunking
  -> Embedding
  -> Hybrid Retrieval
  -> Evidence Export
  -> ChatGPT Plus Batch Package
  -> Manual GPT-4o Scoring in ChatGPT Plus
  -> Import Scores
  -> ESG Index
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.pipeline
```

By default, the pipeline processes every supported report in `data/raw_reports/` inside one Python process.

To run one file for a smoke test:

```powershell
python -m src.pipeline --include-glob AME_2023.pdf --limit 1
```

The pipeline writes upload-ready evidence files:

- `outputs/evidence_dataset.csv`
- `outputs/chatgpt_batches/chatgpt_batch_001.csv`
- `outputs/chatgpt_batches_json/chatgpt_batch_001.json`
- `outputs/chatgpt_prompt_package/chatgpt_batch_001.csv`
- `outputs/chatgpt_prompt_package/chatgpt_batch_001_prompt.md`
- `outputs/chatgpt_batch_manifest.csv`

For scanned PDFs, install OCR extras and run with `--ocr`:

```powershell
pip install -r requirements-ocr.txt
python -m src.pipeline --input-dir data/raw_reports --include-glob ARM_2025.pdf --ocr
```

OCR also requires Tesseract OCR and Poppler binaries on `PATH`.

## Corpus-Scale Execution

The production workflow is optimized for hundreds or thousands of reports:

```text
Load Corpus Once
  -> Parse All Reports
  -> OCR Only When Needed
  -> Cache Parsed Reports
  -> Chunk All Reports
  -> Cache Chunks
  -> Embed New/Changed Reports
  -> Build/Persist One Vector Store
  -> Load Reranker Once
  -> Retrieve Evidence
  -> Cache Evidence
  -> Export ChatGPT Plus Batches
```

Useful flags:

```powershell
python -m src.pipeline --resume
python -m src.pipeline --incremental
python -m src.pipeline --rebuild-parsed
python -m src.pipeline --rebuild-chunks
python -m src.pipeline --rebuild-embeddings
python -m src.pipeline --rebuild-retrieval
python -m src.pipeline --ocr-workers 7 --retrieval-workers 4
python -m src.pipeline --retrieval-mode bm25
python -m src.pipeline --retrieval-mode hybrid
python -m src.pipeline --retrieval-mode hybrid-rerank --rerank-threshold 0.95
```

`--incremental` detects new or modified reports using SHA256 hashes. `--resume` skips stages already marked complete in `outputs/progress_manifest.parquet`.

## Cache And Recovery

The pipeline stores resumable artifacts:

```text
data/parsed_reports/<report>.parquet
data/chunks/<report>_chunks.parquet
data/embeddings/<report>_embeddings.npy
outputs/evidence_warehouse/<company>_<year>_<pillar>_<hash>.parquet
outputs/evidence_cache/<company>_<year>_<indicator>.parquet
outputs/progress_manifest.parquet
outputs/errors.csv
logs/pipeline.log
data/esg_pipeline.duckdb
```

If one report fails, the pipeline logs the error and continues. Failures are written to `outputs/errors.csv` with:

```text
report
stage
error_type
message
traceback
```

Parquet support requires `pyarrow`, which is included in `requirements.txt`.

## Retrieval System

Retrieval is a two-stage system:

1. Build reusable evidence warehouses per report and pillar:
   - `environment_evidence`
   - `social_evidence`
   - `governance_evidence`
2. Reuse the warehouse pools across all indicators in that pillar.

This avoids repeating expensive retrieval for indicators that share the same semantic domain.

Supported retrieval modes:

```powershell
python -m src.pipeline --retrieval-mode bm25
python -m src.pipeline --retrieval-mode hybrid
python -m src.pipeline --retrieval-mode hybrid-rerank
```

`hybrid-rerank` supports reranker skipping:

```powershell
python -m src.pipeline --rerank-threshold 0.95
```

If the retrieval confidence is already above the threshold, the transformer reranker is skipped. Otherwise, candidate sets are reranked in batches per firm-year.

Retrieval diagnostics are written to:

```text
outputs/retrieval_runtime.csv
outputs/evidence_cache/
outputs/evidence_warehouse/
outputs/experiments/
```

`outputs/retrieval_runtime.csv` includes:

```text
report
indicator
runtime_seconds
retrieval_score
rerank_score
reranker_used
```

Each run also writes an experiment record in `outputs/experiments/` with:

```text
experiment_id
embedding_model
reranker_model
chunk_size
top_k
retrieval_mode
runtime
timestamp
```

## Required Codebook

Place the codebook here:

```text
indicators/esg_codebook_vn_50_indicators.csv
```

Required columns:

```text
indicator_id
pillar
category_vi
indicator_name_vi
framework
retrieval_query
keywords_vi
score_0
score_1
score_2
score_3
evidence_requirement
```

Optional but recommended:

```text
definition
```

If `definition` is missing, the pipeline builds a fallback definition from `indicator_name_vi`, `retrieval_query`, `keywords_vi`, and `evidence_requirement`.

## Evidence Dataset

`outputs/evidence_dataset.csv` contains one row per firm-year-indicator:

```text
company
year
indicator_id
pillar
indicator_name
definition
framework
retrieval_query
evidence
page_numbers
evidence_length
chunk_count
page_count
evidence_quality
```

`evidence_quality` is set to `LOW_EVIDENCE` when fewer than two chunks are retrieved after deduplication.

## ChatGPT Plus Batch Workflow

Each batch has a CSV and a matching prompt:

```text
outputs/chatgpt_prompt_package/chatgpt_batch_001.csv
outputs/chatgpt_prompt_package/chatgpt_batch_001_prompt.md
```

Upload both files into ChatGPT Plus and ask it to score the rows using the prompt. The prompt requests a CSV with:

```text
company
year
indicator_id
score
confidence
reasoning
```

Expected scoring rubric:

```text
0 = No disclosure
1 = Qualitative disclosure
2 = Quantitative disclosure
3 = Quantitative disclosure with targets or outcomes
```

## Import Scores

After ChatGPT Plus returns a score CSV, import it and aggregate ESG scores:

```powershell
python -m src.import_scores outputs/manual_scores/chatgpt_scores.csv
```

You can also pass a directory containing multiple returned CSV files:

```powershell
python -m src.import_scores outputs/manual_scores
```

This writes:

- `outputs/indicator_scores/indicator_scores.csv`
- `outputs/ESG_scores/esg_scores.csv`

The importer validates:

- `score` must be one of `0, 1, 2, 3`
- `confidence` must be in `[0, 1]`
- `indicator_id` must exist in the codebook

## Report Naming

Reports should use:

```text
FIRM_YEAR.pdf
```

Examples:

- `AME_2025.pdf`
- `ARM_2024.pdf`

## Pipeline Stages

1. `src/parser.py` extracts page-level text from PDF, TXT, DOCX, HTML, and MD files.
2. `src/chunker.py` splits parsed pages into overlapping chunks.
3. `src/embedder.py` creates deterministic local embeddings.
4. `src/retriever.py` performs hybrid retrieval: BM25 + embedding search + reranking.
5. `src/export_evidence.py` exports deduplicated evidence records.
6. `src/batch_builder.py` creates CSV and JSON batches.
7. `src/prompt_generator.py` creates one prompt per batch.
8. `src/import_scores.py` imports manual ChatGPT Plus scores.
9. `src/aggregator.py` computes E, S, G, and ESG indices from imported scores.

## Directory Layout

```text
data/
  raw_reports/
  parsed_reports/
  chunks/
  embeddings/
indicators/
  esg_codebook_vn_50_indicators.csv
outputs/
  evidence_dataset.csv
  chatgpt_batches/
  chatgpt_batches_json/
  chatgpt_prompt_package/
  chatgpt_batch_manifest.csv
  indicator_scores/
  ESG_scores/
src/
  aggregator.py
  batch_builder.py
  chunker.py
  config.py
  embedder.py
  evidence_builder.py
  export_evidence.py
  import_scores.py
  parser.py
  pipeline.py
  prompt_generator.py
  retriever.py
  reranker.py
vectorstore/
  faiss/
```

## Notes

- The pipeline does not call any LLM API.
- `src/scorer.py` is intentionally disabled for this workflow.
- Manual scoring happens in ChatGPT Plus using the generated batch CSV and prompt markdown files.
