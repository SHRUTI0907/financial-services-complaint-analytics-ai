from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from src.config import (
    ENTITY_MAP_PARQUET_PATH,
    SEC_COMPANY_FACTS_DIR,
    SEC_LINEAGE_PATH,
    SEC_SCALE_METRICS_PATH,
    ensure_directories,
)

SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

CONCEPT_CANDIDATES = {
    "assets": ["Assets"],
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "InterestAndDividendIncomeOperating",
        "InterestIncomeExpenseAfterProvisionForLoanLoss",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_income": ["OperatingIncomeLoss", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"],
}


def _declared_user_agent(user_agent: str | None = None) -> str:
    value = user_agent or os.getenv("SEC_USER_AGENT", "").strip()
    if not value or "@" not in value:
        raise ValueError("Set SEC_USER_AGENT to a descriptive value with contact email before live SEC requests.")
    return value


def fetch_companyfacts(cik: str, user_agent: str, cache_dir: Path = SEC_COMPANY_FACTS_DIR, sleep_seconds: float = 0.12) -> dict:
    ensure_directories()
    padded = str(cik).zfill(10)
    cache_path = cache_dir / f"CIK{padded}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    req = Request(
        SEC_COMPANYFACTS_URL.format(cik=padded),
        headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
    )
    with urlopen(req, timeout=45) as response:
        data = json.loads(response.read().decode("utf-8"))
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(sleep_seconds)
    return data


def _fact_rows(companyfacts: dict, metric_name: str, concepts: list[str]) -> list[dict[str, object]]:
    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    rows = []
    for concept_priority, concept in enumerate(concepts, start=1):
        concept_data = facts.get(concept, {})
        for unit, values in concept_data.get("units", {}).items():
            if unit != "USD":
                continue
            for fact in values:
                if fact.get("val") is None or fact.get("form") not in {"10-K", "10-Q"}:
                    continue
                fy = fact.get("fy")
                fp = str(fact.get("fp", ""))
                if fy is None:
                    continue
                rows.append(
                    {
                        "metric_name": metric_name,
                        "concept_priority": concept_priority,
                        "xbrl_concept": concept,
                        "unit": unit,
                        "fiscal_year": int(fy),
                        "fiscal_period": fp,
                        "period_end": fact.get("end"),
                        "form": fact.get("form"),
                        "filed_date": fact.get("filed"),
                        "accession": fact.get("accn"),
                        "value": float(fact.get("val")),
                    }
                )
    return rows


def resolve_financial_facts(companyfacts: dict, retrieved_at_utc: str | None = None) -> pd.DataFrame:
    retrieved_at_utc = retrieved_at_utc or datetime.now(UTC).isoformat()
    entity_name = companyfacts.get("entityName")
    cik = str(companyfacts.get("cik", "")).zfill(10)
    rows = []
    for metric, concepts in CONCEPT_CANDIDATES.items():
        rows.extend(_fact_rows(companyfacts, metric, concepts))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["company"] = entity_name
    df["sec_cik"] = cik
    df["retrieval_date"] = retrieved_at_utc
    annual = df[df["fiscal_period"].astype(str).str.upper().isin({"FY"})].copy()
    if annual.empty:
        annual = df[df["form"].eq("10-K")].copy()
    annual = annual.sort_values(["metric_name", "fiscal_year", "concept_priority", "filed_date"])
    resolved = annual.groupby(["sec_cik", "company", "metric_name", "fiscal_year"], as_index=False).tail(1)
    return resolved[
        [
            "company",
            "sec_cik",
            "fiscal_year",
            "fiscal_period",
            "metric_name",
            "form",
            "filed_date",
            "xbrl_concept",
            "unit",
            "value",
            "accession",
            "retrieval_date",
        ]
    ].sort_values(["company", "fiscal_year", "metric_name"])


def latest_scale_table(lineage: pd.DataFrame, entity_map: pd.DataFrame) -> pd.DataFrame:
    if lineage.empty:
        return pd.DataFrame()
    latest = lineage.sort_values(["sec_cik", "metric_name", "fiscal_year", "filed_date"]).groupby(["sec_cik", "metric_name"], as_index=False).tail(1)
    wide = latest.pivot_table(index="sec_cik", columns="metric_name", values="value", aggfunc="last").reset_index()
    year = latest.groupby("sec_cik")["fiscal_year"].max().reset_index(name="scale_fiscal_year")
    joined = entity_map.merge(wide, on="sec_cik", how="left").merge(year, on="sec_cik", how="left")
    joined["normalization_allowed"] = (
        joined["confidence_tier"].eq("HIGH CONFIDENCE")
        & joined[["assets", "revenue"]].notna().any(axis=1)
        & ~joined["manual_review_required"].fillna(True).astype(bool)
    )
    joined["normalization_note"] = "Assets/revenue are scale proxies and are not equivalent to customer count or market share."
    return joined


def build_sec_scale_metrics(
    entity_map_path: Path = ENTITY_MAP_PARQUET_PATH,
    lineage_path: Path = SEC_LINEAGE_PATH,
    output_path: Path = SEC_SCALE_METRICS_PATH,
    user_agent: str | None = None,
    max_companies: int | None = None,
) -> dict[str, object]:
    ensure_directories()
    ua = _declared_user_agent(user_agent)
    entity_map = pd.read_parquet(entity_map_path)
    accepted = entity_map[entity_map["confidence_tier"].eq("HIGH CONFIDENCE")].copy()
    if max_companies:
        accepted = accepted.head(max_companies)
    all_lineage = []
    errors = []
    for row in accepted.itertuples():
        try:
            facts = fetch_companyfacts(row.sec_cik, ua)
            resolved = resolve_financial_facts(facts)
            if not resolved.empty:
                resolved["cfpb_company_name"] = row.cfpb_company_name
                resolved["canonical_company_name"] = row.canonical_company_name
                resolved["ticker"] = row.ticker
                all_lineage.append(resolved)
        except Exception as exc:
            errors.append({"cfpb_company_name": row.cfpb_company_name, "sec_cik": row.sec_cik, "error": str(exc)})
    lineage = pd.concat(all_lineage, ignore_index=True) if all_lineage else pd.DataFrame()
    lineage.to_csv(lineage_path, index=False)
    scale = latest_scale_table(lineage, entity_map) if not lineage.empty else pd.DataFrame()
    scale.to_csv(output_path, index=False)
    status = {
        "source": "SEC EDGAR Company Facts API",
        "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "companies_requested": int(len(accepted)),
        "sec_registrants_with_observations": int(lineage["sec_cik"].nunique()) if not lineage.empty else 0,
        "company_year_observations": int(lineage[["sec_cik", "fiscal_year"]].drop_duplicates().shape[0]) if not lineage.empty else 0,
        "errors": errors,
        "lineage_path": str(lineage_path),
        "scale_metrics_path": str(output_path),
    }
    (output_path.with_suffix(".json")).write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch official SEC EDGAR Company Facts scale metrics for mapped CFPB companies.")
    parser.add_argument("--entity-map", type=Path, default=ENTITY_MAP_PARQUET_PATH)
    parser.add_argument("--lineage", type=Path, default=SEC_LINEAGE_PATH)
    parser.add_argument("--output", type=Path, default=SEC_SCALE_METRICS_PATH)
    parser.add_argument("--user-agent", default=os.getenv("SEC_USER_AGENT"))
    parser.add_argument("--max-companies", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(build_sec_scale_metrics(args.entity_map, args.lineage, args.output, args.user_agent, args.max_companies), indent=2))


if __name__ == "__main__":
    main()
