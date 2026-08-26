from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from src.config import COMPLAINTS_PARQUET, METADATA_PATH, ensure_directories
from src.ingestion.cfpb import normalize_columns

CFPB_SEARCH_API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"


def fetch_cfpb_api_window(date_received_min: date, date_received_max: date | None = None, page_size: int = 1000, max_pages: int | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    offset = 0
    pages = 0
    while True:
        params = {
            "date_received_min": date_received_min.isoformat(),
            "size": page_size,
            "from": offset,
            "format": "json",
            "no_aggs": "true",
        }
        if date_received_max is not None:
            params["date_received_max"] = date_received_max.isoformat()
        with urlopen(f"{CFPB_SEARCH_API}?{urlencode(params)}", timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
        hits = data.get("hits", {})
        hit_rows = hits.get("hits", []) if isinstance(hits, dict) else []
        if not hit_rows:
            break
        for hit in hit_rows:
            source = hit.get("_source", {})
            rows.append(
                {
                    "Date received": source.get("date_received"),
                    "Product": source.get("product"),
                    "Sub-product": source.get("sub_product"),
                    "Issue": source.get("issue"),
                    "Sub-issue": source.get("sub_issue"),
                    "Consumer complaint narrative": source.get("complaint_what_happened"),
                    "Company public response": source.get("company_public_response"),
                    "Company": source.get("company"),
                    "State": source.get("state"),
                    "ZIP code": source.get("zip_code"),
                    "Tags": source.get("tags"),
                    "Consumer consent provided?": source.get("consumer_consent_provided"),
                    "Submitted via": source.get("submitted_via"),
                    "Date sent to company": source.get("date_sent_to_company"),
                    "Company response to consumer": source.get("company_response"),
                    "Timely response?": source.get("timely"),
                    "Consumer disputed?": source.get("consumer_disputed"),
                    "Complaint ID": source.get("complaint_id"),
                }
            )
        pages += 1
        offset += page_size
        if max_pages is not None and pages >= max_pages:
            break
    return normalize_columns(pd.DataFrame(rows)) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a CFPB API date window for incremental refresh experiments.")
    parser.add_argument("--date-min", required=True)
    parser.add_argument("--date-max")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--output", type=Path, default=COMPLAINTS_PARQUET)
    args = parser.parse_args()
    df = fetch_cfpb_api_window(date.fromisoformat(args.date_min), date.fromisoformat(args.date_max) if args.date_max else None, args.page_size, args.max_pages)
    if not df.empty:
        ensure_directories()
        df.to_parquet(args.output, index=False)
        METADATA_PATH.write_text(
            json.dumps(
                {
                    "source": "CFPB Consumer Complaint Database API",
                    "source_url": CFPB_SEARCH_API,
                    "date_received_min": args.date_min,
                    "date_received_max": args.date_max,
                    "record_count": int(len(df)),
                    "output_path": str(args.output),
                    "limited_extract": args.max_pages is not None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(df.head().to_string(index=False))
    print(f"rows={len(df):,}")


if __name__ == "__main__":
    main()
