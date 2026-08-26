from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import METADATA_PATH


def load_metadata(path: Path = METADATA_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def lineage_table() -> pd.DataFrame:
    metadata = load_metadata()
    rows = [
        {
            "metric": "Complaint volume",
            "source": "CFPB Consumer Complaint Database",
            "original_fields": "Complaint ID, Date received",
            "transformation": "Count distinct complaint_id by selected dimensions and dates.",
            "refresh": "User-triggered from official CFPB source.",
            "retrieved_at": metadata.get("retrieved_at_utc"),
        },
        {
            "metric": "Narrative topics",
            "source": "CFPB public narratives",
            "original_fields": "Consumer complaint narrative, product, issue, company",
            "transformation": "Clean redaction artifacts, TF-IDF vectorization, NMF topic discovery.",
            "refresh": "Recomputed when NLP pipeline is run.",
            "retrieved_at": metadata.get("retrieved_at_utc"),
        },
        {
            "metric": "Normalized complaints",
            "source": "CFPB + SEC EDGAR companyfacts",
            "original_fields": "Complaint count, assets, revenue",
            "transformation": "Complaints divided by assets/revenue in billions for defensibly matched companies.",
            "refresh": "SEC refresh user-triggered; denominator caveats shown.",
            "retrieved_at": metadata.get("retrieved_at_utc"),
        },
        {
            "metric": "Estimated annual capacity value",
            "source": "CFPB observed volume + BLS wage benchmark + user assumptions",
            "original_fields": "Complaint count, OEWS wage, selected operational assumptions",
            "transformation": "Observed volume * handling time * addressable share * time reduction * adoption * success probability * wage.",
            "refresh": "Interactive calculation.",
            "retrieved_at": metadata.get("retrieved_at_utc"),
        },
    ]
    return pd.DataFrame(rows)
