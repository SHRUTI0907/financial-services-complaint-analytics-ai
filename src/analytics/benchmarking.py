from __future__ import annotations

import pandas as pd


def company_benchmark(complaints: pd.DataFrame, scale_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    data = complaints.copy()
    data["year"] = pd.to_datetime(data["date_received"], errors="coerce").dt.year
    base = complaints.groupby("company", dropna=False).agg(
        raw_complaints=("complaint_id", "count"),
        products=("product", "nunique"),
        narrative_count=("consumer_complaint_narrative", lambda s: int(s.notna().sum())),
        timely_rate=("timely_response", lambda s: float((s.astype(str).str.lower() == "yes").mean())),
    ).reset_index()
    if scale_metrics is None or scale_metrics.empty:
        base["normalization_note"] = "No SEC scale denominator matched."
        return base.sort_values("raw_complaints", ascending=False)
    map_cols = [
        col
        for col in [
            "cfpb_company_name",
            "canonical_company_name",
            "sec_cik",
            "ticker",
            "confidence_tier",
            "manual_review_required",
            "scale_fiscal_year",
            "assets",
            "revenue",
            "normalization_allowed",
        ]
        if col in scale_metrics.columns
    ]
    joined = base.merge(scale_metrics[map_cols], left_on="company", right_on="cfpb_company_name", how="left")
    if "normalization_allowed" in joined.columns:
        allowed = joined["normalization_allowed"].fillna(False).astype(bool)
    else:
        allowed = joined["cfpb_company_name"].notna()
    assets = pd.to_numeric(joined.get("assets"), errors="coerce")
    revenue = pd.to_numeric(joined.get("revenue"), errors="coerce")
    joined["complaints_per_1b_assets"] = pd.NA
    joined["complaints_per_1b_revenue"] = pd.NA
    joined.loc[allowed & assets.gt(0), "complaints_per_1b_assets"] = joined.loc[allowed & assets.gt(0), "raw_complaints"] / (assets[allowed & assets.gt(0)] / 1_000_000_000)
    joined.loc[allowed & revenue.gt(0), "complaints_per_1b_revenue"] = joined.loc[allowed & revenue.gt(0), "raw_complaints"] / (revenue[allowed & revenue.gt(0)] / 1_000_000_000)
    joined["normalization_note"] = "Assets/revenue are scale proxies and are not equivalent to customer count or market share."
    return joined.sort_values("raw_complaints", ascending=False)
