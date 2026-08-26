from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import COMPLAINTS_PARQUET, METADATA_PATH, RAW_DIR, ensure_directories

CFPB_BULK_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"

EXPECTED_COLUMNS = {
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "consumer_complaint_narrative",
    "Company public response": "company_public_response",
    "Company": "company",
    "State": "state",
    "ZIP code": "zip_code",
    "Tags": "tags",
    "Consumer consent provided?": "consumer_consent_provided",
    "Submitted via": "submitted_via",
    "Date sent to company": "date_sent_to_company",
    "Company response to consumer": "company_response_to_consumer",
    "Timely response?": "timely_response",
    "Consumer disputed?": "consumer_disputed",
    "Complaint ID": "complaint_id",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = {col: EXPECTED_COLUMNS[col] for col in df.columns if col in EXPECTED_COLUMNS}
    df = df.rename(columns=present)
    for col in EXPECTED_COLUMNS.values():
        if col not in df.columns:
            df[col] = pd.NA
    df["complaint_id"] = pd.to_numeric(df["complaint_id"], errors="coerce").astype("Int64")
    for col in ["date_received", "date_sent_to_company"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    text_cols = [col for col in EXPECTED_COLUMNS.values() if col not in {"complaint_id", "date_received", "date_sent_to_company"}]
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()
    return df[list(EXPECTED_COLUMNS.values())]


def ingest_cfpb_bulk(
    url: str = CFPB_BULK_URL,
    raw_zip_path: Path | None = None,
    output_path: Path = COMPLAINTS_PARQUET,
    max_records: int | None = None,
    chunksize: int = 250_000,
) -> dict[str, object]:
    ensure_directories()
    start = time.perf_counter()
    raw_zip_path = raw_zip_path or RAW_DIR / "complaints.csv.zip"
    if not raw_zip_path.exists():
        urlretrieve(url, raw_zip_path)

    total_rows = 0
    writer: pq.ParquetWriter | None = None
    date_min = None
    date_max = None
    narrative_count = 0
    with ZipFile(raw_zip_path) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("CFPB zip did not contain a CSV file.")
        with zf.open(csv_names[0]) as fh:
            for chunk in pd.read_csv(fh, chunksize=chunksize, low_memory=False):
                normalized = normalize_columns(chunk)
                if max_records is not None:
                    remaining = max_records - total_rows
                    if remaining <= 0:
                        break
                    normalized = normalized.head(remaining)
                if normalized.empty:
                    break
                chunk_date_min = normalized["date_received"].min()
                chunk_date_max = normalized["date_received"].max()
                if pd.notna(chunk_date_min):
                    date_min = chunk_date_min if date_min is None else min(date_min, chunk_date_min)
                if pd.notna(chunk_date_max):
                    date_max = chunk_date_max if date_max is None else max(date_max, chunk_date_max)
                narrative_count += int(normalized["consumer_complaint_narrative"].notna().sum())
                table = pa.Table.from_pandas(normalized, preserve_index=False)
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
                writer.write_table(table)
                total_rows += len(normalized)
                if max_records is not None and total_rows >= max_records:
                    break

    if writer is None:
        raise ValueError("No CFPB complaint rows were ingested.")
    writer.close()
    metadata = {
        "source": "CFPB Consumer Complaint Database",
        "source_url": url,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "raw_zip_path": str(raw_zip_path),
        "output_path": str(output_path),
        "record_count": int(total_rows),
        "date_min": None if date_min is None else str(date_min.date()),
        "date_max": None if date_max is None else str(date_max.date()),
        "narrative_count": int(narrative_count),
        "runtime_seconds": round(time.perf_counter() - start, 2),
        "limited_extract": max_records is not None,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest official CFPB Consumer Complaint Database into Parquet.")
    parser.add_argument("--url", default=CFPB_BULK_URL)
    parser.add_argument("--output", type=Path, default=COMPLAINTS_PARQUET)
    parser.add_argument("--max-records", type=int, default=None, help="Optional development cap. Omit for full dataset.")
    args = parser.parse_args()
    metadata = ingest_cfpb_bulk(url=args.url, output_path=args.output, max_records=args.max_records)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
