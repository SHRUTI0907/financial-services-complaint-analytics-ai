from __future__ import annotations

import numpy as np
import pandas as pd


def add_month(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date_received"] = pd.to_datetime(out["date_received"], errors="coerce")
    out["month"] = out["date_received"].dt.to_period("M").dt.to_timestamp()
    return out


def kpi_summary(df: pd.DataFrame) -> dict[str, float | int | str | None]:
    dated = add_month(df)
    narrative_col = "consumer_complaint_narrative"
    return {
        "complaints": int(len(df)),
        "companies": int(df["company"].nunique()) if "company" in df else 0,
        "products": int(df["product"].nunique()) if "product" in df else 0,
        "issues": int(df["issue"].nunique()) if "issue" in df else 0,
        "narratives": int(df[narrative_col].notna().sum()) if narrative_col in df else 0,
        "date_min": None if dated["date_received"].isna().all() else str(dated["date_received"].min().date()),
        "date_max": None if dated["date_received"].isna().all() else str(dated["date_received"].max().date()),
    }


def volume_by_month(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    dated = add_month(df).dropna(subset=["month"])
    group_cols = ["month"] + ([group_col] if group_col else [])
    result = dated.groupby(group_cols, dropna=False).size().reset_index(name="complaints")
    result["rolling_3m"] = result.groupby(group_col)["complaints"].transform(lambda s: s.rolling(3, min_periods=1).mean()) if group_col else result["complaints"].rolling(3, min_periods=1).mean()
    return result


def top_categories(df: pd.DataFrame, col: str, n: int = 15) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col, "complaints", "share"])
    counts = df[col].fillna("Missing").value_counts().head(n).reset_index()
    counts.columns = [col, "complaints"]
    counts["share"] = counts["complaints"] / max(len(df), 1)
    return counts


def yoy_growth(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    dated = add_month(df)
    dated["year"] = dated["date_received"].dt.year
    counts = dated.groupby([group_col, "year"], dropna=False).size().reset_index(name="complaints")
    counts["prior_year"] = counts.groupby(group_col)["complaints"].shift(1)
    counts["yoy_growth"] = np.where(counts["prior_year"] > 0, (counts["complaints"] - counts["prior_year"]) / counts["prior_year"], np.nan)
    return counts
