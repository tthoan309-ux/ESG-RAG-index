from __future__ import annotations

from pathlib import Path

import pandas as pd


class DuckDBBackend:
    def __init__(self, path: Path):
        self.path = path

    def available(self) -> bool:
        try:
            import duckdb  # noqa: F401

            return True
        except ImportError:
            return False

    def write_table(self, name: str, frame: pd.DataFrame) -> None:
        if not self.available():
            return
        import duckdb

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.path)) as con:
            con.register("_frame", frame)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _frame")
