from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from esg_prescore.api_scoring import run_api_scoring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional API scoring pilot for an existing pre-score run.")
    parser.add_argument("--run-dir", required=True, help="Existing prescore output directory.")
    parser.add_argument("--output-dir", required=True, help="New or empty output directory for API scoring outputs.")
    parser.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    parser.add_argument("--model", default=None, help="Model override. Defaults to provider-specific pilot model.")
    parser.add_argument("--limit", type=int, default=1, help="Rows to score. Default is one row for cost-safe testing.")
    parser.add_argument("--dry-run", action="store_true", help="Build request payloads without calling the API.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_api_scoring(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        provider=args.provider,
        model=args.model,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
