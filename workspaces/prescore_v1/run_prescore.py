from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from esg_prescore.pipeline import run_prescore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build evidence packages and stop before ESG scoring.")
    parser.add_argument("--corpus", required=True, help="ESG source corpus parquet exported by financialdistress.")
    parser.add_argument("--output-dir", required=True, help="New or empty output directory.")
    parser.add_argument("--config", default=str(WORKSPACE_ROOT / "config" / "prescore.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_prescore(args.corpus, args.output_dir, args.config, WORKSPACE_ROOT)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
