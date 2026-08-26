from __future__ import annotations

import re

import pandas as pd

REDACTION_PATTERN = re.compile(r"\bX{2,}\b|XXXX+|xx/xx/\d{2,4}", flags=re.IGNORECASE)
SPACE_PATTERN = re.compile(r"\s+")


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = REDACTION_PATTERN.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = SPACE_PATTERN.sub(" ", text).strip()
    return text


def narrative_frame(df: pd.DataFrame, min_chars: int = 80) -> pd.DataFrame:
    out = df.copy()
    out["clean_narrative"] = out["consumer_complaint_narrative"].map(clean_text)
    return out[out["clean_narrative"].str.len() >= min_chars].copy()
