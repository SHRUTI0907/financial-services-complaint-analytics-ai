from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import COMPLAINTS_PARQUET, DUCKDB_PATH

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised in minimal environments
    duckdb = None


@dataclass
class StoreStatus:
    available: bool
    message: str
    complaint_path: Path


def get_store_status(path: Path = COMPLAINTS_PARQUET) -> StoreStatus:
    if not path.exists():
        return StoreStatus(False, "No processed CFPB complaint store found. Run the ingestion pipeline first.", path)
    return StoreStatus(True, "Processed CFPB complaint store is available.", path)


def query(sql: str, params: dict[str, Any] | None = None, parquet_path: Path = COMPLAINTS_PARQUET) -> pd.DataFrame:
    if duckdb is None:
        raise RuntimeError("duckdb is not installed. Install requirements.txt to query the analytical store.")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing complaint parquet store: {parquet_path}")
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("CREATE OR REPLACE VIEW complaints AS SELECT * FROM read_parquet(?)", [str(parquet_path)])
    try:
        return con.execute(sql, params or {}).df()
    finally:
        con.close()


def read_complaints_sample(n: int = 10000, parquet_path: Path = COMPLAINTS_PARQUET) -> pd.DataFrame:
    return query(f"SELECT * FROM complaints LIMIT {int(n)}", parquet_path=parquet_path)
