from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import PipelinePaths, RetrievalConfig
from .export_evidence import load_codebook
from .manifest_manager import ManifestManager, sha256_file
from .ontology import OntologyManager
from .retrieval_manager import retrieval_config_hash
from .storage import read_table, write_table


def migrate_legacy_indicator_cache(root: Path | None = None) -> pd.DataFrame:
    paths = PipelinePaths(root=root or PipelinePaths().root)
    codebook = OntologyManager(paths.indicator_ontology).attach(load_codebook(paths.indicators))
    manifest = ManifestManager(paths.progress_manifest)
    config_hash = retrieval_config_hash(RetrievalConfig())
    rows: list[dict] = []

    for report in sorted(paths.raw_reports.glob("*.pdf")):
        report_id = report.stem
        report_hash = _report_hash(report_id, report, manifest)
        company, year = _company_year(report_id)
        for group_id, group in codebook.groupby(["domain", "subdomain"]):
            ontology_group = "_".join(str(part).lower().replace(" ", "_").replace("/", "_") for part in group_id)
            warehouse_path = paths.evidence_warehouse / f"{report_hash}_{ontology_group}_{config_hash}.parquet"
            if warehouse_path.exists():
                rows.append({"report": report_id, "ontology_group": ontology_group, "status": "exists", "chunks": 0})
                continue
            chunks = _chunks_from_legacy_cache(paths, company, year, group)
            if not chunks:
                rows.append({"report": report_id, "ontology_group": ontology_group, "status": "missing_legacy_cache", "chunks": 0})
                continue
            write_table(warehouse_path, pd.DataFrame(chunks))
            rows.append({"report": report_id, "ontology_group": ontology_group, "status": "migrated", "chunks": len(chunks)})

    report = pd.DataFrame(rows)
    report.to_csv(paths.root / "outputs" / "warehouse_migration_report.csv", index=False)
    return report


def _chunks_from_legacy_cache(paths: PipelinePaths, company: str, year: str, group: pd.DataFrame) -> list[dict]:
    chunks: list[dict] = []
    seen: set[str] = set()
    for _, indicator in group.iterrows():
        legacy_path = paths.evidence_cache / f"{company}_{year}_{indicator.indicator_id}.parquet"
        if not legacy_path.exists():
            continue
        frame = read_table(legacy_path)
        if frame.empty:
            continue
        row = frame.iloc[0].to_dict()
        evidence = str(row.get("evidence", ""))
        if not evidence or evidence in seen:
            continue
        seen.add(evidence)
        chunks.append(
            {
                "chunk_id": f"legacy_{company}_{year}_{indicator.indicator_id}",
                "company": company,
                "year": year,
                "source_file": f"{company}_{year}.pdf",
                "page": str(row.get("page_numbers", "")),
                "text": evidence,
                "retrieval_score": float(row.get("retrieval_score", 0.0) or 0.0),
                "reranker_score": float(row.get("rerank_score", 0.0) or 0.0),
                "reranker_model": "legacy-indicator-cache-migrated-to-topic",
            }
        )
    return chunks


def _report_hash(report_id: str, report: Path, manifest: ManifestManager) -> str:
    row = manifest.frame.loc[manifest.frame["report_id"] == report_id]
    if not row.empty:
        return str(row.iloc[0]["file_hash"])
    return sha256_file(report)


def _company_year(report_id: str) -> tuple[str, str]:
    if "_" not in report_id:
        return report_id, ""
    company, year = report_id.rsplit("_", 1)
    return company, year


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy indicator evidence cache into topic warehouse cache.")
    parser.add_argument("--root", type=Path, default=PipelinePaths().root)
    args = parser.parse_args()
    frame = migrate_legacy_indicator_cache(args.root)
    print(frame["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
