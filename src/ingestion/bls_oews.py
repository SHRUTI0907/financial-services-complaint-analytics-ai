from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd

from src.config import ARTIFACT_DIR, BLS_STATUS_PATH, BLS_WAGE_BENCHMARKS_PATH, RAW_DIR, ensure_directories

BLS_OEWS_URL = "https://www.bls.gov/oes/special.requests/oesm25nat.zip"
BLS_BENCHMARK_PERIOD = "May 2025"
DEFAULT_SOC_CODES = {
    "43-4051": "Customer Service Representatives",
    "43-9061": "Office Clerks, General",
    "13-1111": "Management Analysts",
    "13-2099": "Financial Specialists, All Other",
    "15-2051": "Data Scientists",
}


def parse_oews_workbook(df: pd.DataFrame, soc_codes: set[str] | None = None) -> pd.DataFrame:
    soc_codes = soc_codes or set(DEFAULT_SOC_CODES)
    data = df.copy()
    data.columns = [str(col).strip().lower() for col in data.columns]
    occ_col = "occ_code" if "occ_code" in data.columns else "o_group"
    filtered = data[data[occ_col].astype(str).isin(soc_codes)].copy()
    for col in ["h_mean", "a_mean", "h_median", "a_median"]:
        if col in filtered.columns:
            filtered[col] = pd.to_numeric(filtered[col].astype(str).str.replace(",", "", regex=False).str.replace("*", "", regex=False), errors="coerce")
    keep = [col for col in ["occ_code", "occ_title", "h_mean", "a_mean", "h_median", "a_median"] if col in filtered.columns]
    result = filtered[keep].rename(
        columns={
            "occ_code": "occupation_code",
            "occ_title": "occupation",
            "h_mean": "mean_hourly_wage",
            "a_mean": "mean_annual_wage",
            "h_median": "median_hourly_wage",
            "a_median": "median_annual_wage",
        }
    )
    result["benchmark_period"] = BLS_BENCHMARK_PERIOD
    result["source"] = "U.S. Bureau of Labor Statistics OEWS national estimates"
    result["source_url"] = BLS_OEWS_URL
    result["retrieval_date"] = datetime.now(UTC).isoformat()
    result["benchmark_note"] = "External labor-cost benchmark only; not company payroll data."
    return result.sort_values("occupation_code").reset_index(drop=True)


def ingest_bls_oews(
    url: str = BLS_OEWS_URL,
    output_path: Path = BLS_WAGE_BENCHMARKS_PATH,
    soc_codes: set[str] | None = None,
) -> pd.DataFrame:
    ensure_directories()
    zip_path = RAW_DIR / "bls_oews_national.zip"
    if not zip_path.exists():
        urlretrieve(url, zip_path)
    with ZipFile(zip_path) as zf:
        names = [name for name in zf.namelist() if name.lower().endswith((".xlsx", ".xls"))]
        if not names:
            raise ValueError("BLS OEWS zip did not contain an Excel file.")
        with zf.open(names[0]) as fh:
            df = pd.read_excel(fh)
    result = parse_oews_workbook(df, soc_codes=soc_codes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    BLS_STATUS_PATH.write_text(
        json.dumps(
            {
                "source": "U.S. Bureau of Labor Statistics OEWS",
                "source_url": url,
                "benchmark_period": BLS_BENCHMARK_PERIOD,
                "occupations": int(len(result)),
                "output_path": str(output_path),
                "retrieved_at_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest official BLS OEWS national wage benchmarks.")
    parser.add_argument("--url", default=BLS_OEWS_URL)
    parser.add_argument("--output", type=Path, default=BLS_WAGE_BENCHMARKS_PATH)
    args = parser.parse_args()
    print(ingest_bls_oews(args.url, args.output).to_string(index=False))


if __name__ == "__main__":
    main()
