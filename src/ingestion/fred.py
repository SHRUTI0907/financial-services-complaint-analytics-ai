from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from src.config import COMPLAINTS_PARQUET, FRED_ANALYSIS_PATH, FRED_OBSERVATIONS_PATH, ensure_directories

FRED_SERIES = {
    "REVOLSL": "Consumer revolving credit owned and securitized, outstanding",
    "TERMCBCCALLNS": "Commercial bank credit card plans, all accounts, finance rate",
    "DRCCLACBS": "Delinquency rate on credit card loans, all commercial banks",
    "UNRATE": "Civilian unemployment rate",
}


def fetch_fred_series(series_id: str, api_key: str, observation_start: str = "2016-01-01") -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
    }
    url = "https://api.stlouisfed.org/fred/series/observations?" + urlencode(params)
    with urlopen(url, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = []
    for obs in payload.get("observations", []):
        value = pd.to_numeric(obs.get("value"), errors="coerce")
        if pd.isna(value):
            continue
        rows.append(
            {
                "series_id": series_id,
                "description": FRED_SERIES.get(series_id, series_id),
                "date": obs.get("date"),
                "value": float(value),
                "retrieval_date": datetime.now(UTC).isoformat(),
                "source": "Federal Reserve Bank of St. Louis FRED",
            }
        )
    return pd.DataFrame(rows)


def ingest_fred(series_ids: list[str] | None = None, api_key: str | None = None, output_path: Path = FRED_OBSERVATIONS_PATH) -> dict[str, object]:
    ensure_directories()
    api_key = api_key or os.getenv("FRED_API_KEY")
    if not api_key:
        status = {
            "enabled": False,
            "reason": "FRED_API_KEY is not configured.",
            "source": "Federal Reserve Bank of St. Louis FRED",
            "output_path": str(output_path),
        }
        output_path.with_suffix(".json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status
    frames = [fetch_fred_series(series_id, api_key) for series_id in (series_ids or list(FRED_SERIES))]
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    data.to_csv(output_path, index=False)
    return {
        "enabled": True,
        "series": int(data["series_id"].nunique()) if not data.empty else 0,
        "observations": int(len(data)),
        "output_path": str(output_path),
    }


def analyze_macro_context(
    complaints_path: Path = COMPLAINTS_PARQUET,
    observations_path: Path = FRED_OBSERVATIONS_PATH,
    output_path: Path = FRED_ANALYSIS_PATH,
) -> pd.DataFrame:
    complaints = pd.read_parquet(complaints_path)
    if not observations_path.exists():
        return pd.DataFrame()
    obs = pd.read_csv(observations_path)
    if obs.empty:
        return pd.DataFrame()
    complaints["month"] = pd.to_datetime(complaints["date_received"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    monthly = complaints.groupby("month").size().reset_index(name="complaints")
    obs["month"] = pd.to_datetime(obs["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    rows = []
    for series_id, group in obs.groupby("series_id"):
        merged = monthly.merge(group.groupby("month")["value"].mean().reset_index(), on="month", how="inner").dropna()
        if len(merged) < 12:
            continue
        for lag in [0, 1, 3, 6]:
            shifted = merged.copy()
            shifted["macro_lagged"] = shifted["value"].shift(lag)
            corr = shifted[["complaints", "macro_lagged"]].dropna().corr().iloc[0, 1]
            rows.append(
                {
                    "series_id": series_id,
                    "description": FRED_SERIES.get(series_id, series_id),
                    "lag_months": lag,
                    "correlation": None if pd.isna(corr) else float(corr),
                    "paired_months": int(shifted[["complaints", "macro_lagged"]].dropna().shape[0]),
                    "caveat": "Correlation does not imply causation.",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest optional official FRED macro series and analyze complaint trend context.")
    parser.add_argument("--series", nargs="*", default=list(FRED_SERIES))
    args = parser.parse_args()
    print(json.dumps(ingest_fred(args.series), indent=2))
    print(analyze_macro_context().to_string(index=False))


if __name__ == "__main__":
    main()
