from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from src.analytics.anomaly import emerging_issue_score
from src.analytics.complaints import add_month, top_categories, volume_by_month
from src.rag.retrieval import RagFilters


def apply_filters(df: pd.DataFrame, filters: RagFilters | None = None) -> pd.DataFrame:
    if filters is None:
        return df.copy()
    out = df.copy()
    for col in ["company", "product", "issue", "state"]:
        value = getattr(filters, col)
        if value:
            out = out[out[col].astype(str).str.lower() == str(value).lower()]
    if filters.date_start:
        dates = pd.to_datetime(out["date_received"], errors="coerce")
        out = out[dates >= pd.Timestamp(filters.date_start)]
    if filters.date_end:
        dates = pd.to_datetime(out["date_received"], errors="coerce")
        out = out[dates <= pd.Timestamp(filters.date_end)]
    return out


def get_complaint_volume(df: pd.DataFrame, filters: RagFilters | None = None) -> dict[str, object]:
    subset = apply_filters(df, filters)
    dates = pd.to_datetime(subset["date_received"], errors="coerce")
    return {
        "metric_id": "complaint_volume",
        "filters": asdict(filters) if filters else {},
        "complaints": int(len(subset)),
        "date_min": None if dates.dropna().empty else str(dates.min().date()),
        "date_max": None if dates.dropna().empty else str(dates.max().date()),
        "citation": "[Analytics: complaint volume]",
    }


def get_top_issues(df: pd.DataFrame, filters: RagFilters | None = None, n: int = 5) -> dict[str, object]:
    subset = apply_filters(df, filters)
    rows = top_categories(subset, "issue", n).to_dict("records") if not subset.empty else []
    return {"metric_id": "top_issues", "rows": rows, "citation": "[Analytics: top issues]"}


def get_growth_rate(df: pd.DataFrame, filters: RagFilters | None = None, months: int = 6) -> dict[str, object]:
    subset = apply_filters(df, filters)
    monthly = volume_by_month(subset)
    if len(monthly) < months * 2:
        return {"metric_id": "growth_rate", "status": "insufficient_history", "citation": "[Analytics: recent complaint growth]"}
    monthly = monthly.sort_values("month")
    recent = monthly.tail(months)["complaints"].sum()
    prior = monthly.iloc[-months * 2 : -months]["complaints"].sum()
    growth = None if prior == 0 else (recent - prior) / prior
    return {
        "metric_id": "growth_rate",
        "months": months,
        "recent_complaints": int(recent),
        "prior_complaints": int(prior),
        "growth_rate": None if growth is None else float(growth),
        "citation": "[Analytics: recent complaint growth]",
    }


def get_response_metrics(df: pd.DataFrame, filters: RagFilters | None = None) -> dict[str, object]:
    subset = apply_filters(df, filters)
    if subset.empty:
        return {"metric_id": "response_metrics", "status": "no_records", "citation": "[Analytics: response metrics]"}
    timely = subset["timely_response"].astype(str).str.lower().eq("yes").mean()
    responses = subset["company_response_to_consumer"].fillna("Missing").value_counts().head(5).to_dict()
    return {"metric_id": "response_metrics", "timely_response_rate": float(timely), "top_responses": responses, "citation": "[Analytics: response metrics]"}


def get_anomaly_summary(df: pd.DataFrame, group_col: str = "issue") -> dict[str, object]:
    emerg = emerging_issue_score(df, group_col).head(5)
    rows = emerg.replace({np.nan: None}).to_dict("records")
    return {"metric_id": "anomaly_summary", "group_col": group_col, "rows": rows, "citation": "[Analytics: emerging-risk score]"}


def run_default_analytics(df: pd.DataFrame, filters: RagFilters | None = None) -> list[dict[str, object]]:
    return [
        get_complaint_volume(df, filters),
        get_growth_rate(df, filters),
        get_top_issues(df, filters),
        get_response_metrics(df, filters),
    ]
