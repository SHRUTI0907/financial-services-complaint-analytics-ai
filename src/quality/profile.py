from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "complaint_id",
    "date_received",
    "product",
    "issue",
    "company",
    "state",
    "submitted_via",
    "company_response_to_consumer",
    "timely_response",
    "consumer_complaint_narrative",
}


def validate_complaints_schema(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
    if "complaint_id" in df.columns and df["complaint_id"].duplicated().any():
        errors.append("Duplicate complaint_id values detected.")
    if "date_received" in df.columns:
        dates = pd.to_datetime(df["date_received"], errors="coerce")
        if dates.isna().mean() > 0.01:
            errors.append("More than 1% of date_received values could not be parsed.")
        if not dates.dropna().empty and dates.max() > pd.Timestamp.today() + pd.Timedelta(days=1):
            errors.append("date_received contains future dates.")
    return errors


def build_quality_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = max(len(df), 1)
    for col in df.columns:
        rows.append(
            {
                "field": col,
                "records": len(df),
                "missing_count": int(df[col].isna().sum()),
                "missing_rate": round(float(df[col].isna().sum() / total), 4),
                "unique_values": int(df[col].nunique(dropna=True)),
                "example_non_null": None if df[col].dropna().empty else str(df[col].dropna().iloc[0])[:160],
            }
        )
    return pd.DataFrame(rows)


def response_distribution(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["company_response_to_consumer", "timely_response", "submitted_via"]
    frames = []
    for col in cols:
        if col in df.columns:
            counts = df[col].fillna("Missing").value_counts(dropna=False).reset_index()
            counts.columns = ["category", "complaints"]
            counts["dimension"] = col
            frames.append(counts[["dimension", "category", "complaints"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["dimension", "category", "complaints"])
