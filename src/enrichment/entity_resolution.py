from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from src.config import COMPLAINTS_PARQUET, ENTITY_MAP_CSV_PATH, ENTITY_MAP_PARQUET_PATH, ENTITY_REVIEW_PATH

LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "llc",
    "l.l.c",
    "ltd",
    "limited",
    "na",
    "n.a",
    "national",
    "association",
    "bank",
    "bancorp",
    "financial",
    "services",
    "of",
    "the",
}

ALIASES = {
    "jpmorgan chase": "JPMorgan Chase & Co.",
    "jp morgan chase": "JPMorgan Chase & Co.",
    "chase bank": "JPMorgan Chase & Co.",
    "capital one": "Capital One Financial Corporation",
    "bank of america": "Bank of America Corporation",
    "wells fargo": "Wells Fargo & Company",
    "citibank": "Citigroup Inc.",
    "citi": "Citigroup Inc.",
    "american express": "American Express Company",
    "discover": "Discover Financial Services",
    "synchrony": "Synchrony Financial",
}


@dataclass(frozen=True)
class MatchDecision:
    match_score: float
    confidence_tier: str
    manual_review_required: bool


def normalize_company_name(name: str) -> str:
    text = re.sub(r"&", " and ", str(name).lower())
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [tok for tok in text.split() if tok not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def confidence_from_score(score: float, method: str) -> MatchDecision:
    if method.startswith("manual") and score >= 0.97:
        return MatchDecision(score, "HIGH CONFIDENCE", False)
    if score >= 0.96:
        return MatchDecision(score, "HIGH CONFIDENCE", False)
    if score >= 0.88:
        return MatchDecision(score, "MEDIUM - REVIEW", True)
    return MatchDecision(score, "LOW - REJECT", True)


def _coerce_seed_map(seed: pd.DataFrame) -> pd.DataFrame:
    out = seed.copy()
    if "confidence" in out.columns and "match_score" not in out.columns:
        out = out.rename(columns={"confidence": "match_score", "manual_review_flag": "manual_review_required"})
    for col in ["cfpb_company_name", "canonical_company_name", "sec_cik", "ticker", "match_method", "match_score", "manual_review_required", "notes"]:
        if col not in out.columns:
            out[col] = pd.NA
    out["match_score"] = pd.to_numeric(out["match_score"], errors="coerce").fillna(0.0)
    decisions = [confidence_from_score(float(row.match_score), str(row.match_method)) for row in out.itertuples()]
    out["confidence_tier"] = [d.confidence_tier for d in decisions]
    out["manual_review_required"] = [bool(d.manual_review_required or str(v).lower() == "true") for d, v in zip(decisions, out["manual_review_required"])]
    out["sec_cik"] = out["sec_cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    return out[
        [
            "cfpb_company_name",
            "canonical_company_name",
            "sec_cik",
            "ticker",
            "match_method",
            "match_score",
            "confidence_tier",
            "manual_review_required",
            "notes",
        ]
    ]


def read_seed_entity_map(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype={"sec_cik": str})
    except pd.errors.ParserError:
        rows = []
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines[1:]:
            cik_match = re.search(r",(\d{10}),", line)
            if not cik_match:
                continue
            left = line[: cik_match.start()]
            right = line[cik_match.end() :]
            parts = right.split(",", 4)
            if len(parts) != 5:
                continue
            cfpb_and_canonical = left.rsplit(",", 1)
            if len(cfpb_and_canonical) != 2:
                continue
            rows.append(
                {
                    "cfpb_company_name": cfpb_and_canonical[0],
                    "canonical_company_name": cfpb_and_canonical[1],
                    "sec_cik": cik_match.group(1),
                    "ticker": parts[0],
                    "match_method": parts[1],
                    "confidence": parts[2],
                    "manual_review_flag": parts[3],
                    "notes": parts[4],
                }
            )
        return pd.DataFrame(rows)


def build_entity_resolution_artifacts(
    complaints_path: Path = COMPLAINTS_PARQUET,
    seed_map_path: Path = ENTITY_MAP_CSV_PATH,
    parquet_path: Path = ENTITY_MAP_PARQUET_PATH,
    review_path: Path = ENTITY_REVIEW_PATH,
    top_n_review: int = 200,
) -> dict[str, object]:
    complaints = pd.read_parquet(complaints_path)
    companies = complaints["company"].dropna().astype(str).value_counts().rename_axis("cfpb_company_name").reset_index(name="complaints")
    seed = _coerce_seed_map(read_seed_entity_map(seed_map_path))
    seed["normalized_cfpb_name"] = seed["cfpb_company_name"].map(normalize_company_name)
    seed["normalized_canonical_name"] = seed["canonical_company_name"].map(normalize_company_name)

    mapped_names = set(seed["cfpb_company_name"].astype(str).str.lower())
    candidates = companies[~companies["cfpb_company_name"].str.lower().isin(mapped_names)].head(top_n_review).copy()
    canonical = seed[["canonical_company_name", "normalized_canonical_name"]].drop_duplicates()
    review_rows = []
    for row in candidates.itertuples():
        norm = normalize_company_name(row.cfpb_company_name)
        alias_target = next((target for alias, target in ALIASES.items() if alias in norm), None)
        if alias_target:
            match = seed[seed["canonical_company_name"].eq(alias_target)].head(1)
            if not match.empty:
                score = 0.90
                review_rows.append(
                    {
                        "cfpb_company_name": row.cfpb_company_name,
                        "complaints": int(row.complaints),
                        "suggested_canonical_company_name": alias_target,
                        "suggested_sec_cik": match.iloc[0]["sec_cik"],
                        "suggested_ticker": match.iloc[0]["ticker"],
                        "match_method": "alias_suggestion",
                        "match_score": score,
                        "confidence_tier": "MEDIUM - REVIEW",
                        "manual_review_required": True,
                        "notes": "Alias suggestion only; do not normalize until reviewed.",
                    }
                )
                continue
        best = None
        for candidate in canonical.itertuples():
            score = SequenceMatcher(None, norm, candidate.normalized_canonical_name).ratio()
            if best is None or score > best[0]:
                best = (score, candidate.canonical_company_name)
        score, name = best if best else (0.0, "")
        decision = confidence_from_score(score, "fuzzy_suggestion")
        review_rows.append(
            {
                "cfpb_company_name": row.cfpb_company_name,
                "complaints": int(row.complaints),
                "suggested_canonical_company_name": name,
                "suggested_sec_cik": pd.NA,
                "suggested_ticker": pd.NA,
                "match_method": "fuzzy_suggestion",
                "match_score": float(score),
                "confidence_tier": decision.confidence_tier,
                "manual_review_required": True,
                "notes": "Suggestion only. Weak matches must not be used for normalization.",
            }
        )
    review = pd.DataFrame(review_rows)
    seed_names = set(seed["cfpb_company_name"].astype(str).str.lower())
    mapped_companies = companies[companies["cfpb_company_name"].astype(str).str.lower().isin(seed_names)]
    high_names = set(seed.loc[seed["confidence_tier"].eq("HIGH CONFIDENCE"), "cfpb_company_name"].astype(str).str.lower())
    high_companies = companies[companies["cfpb_company_name"].astype(str).str.lower().isin(high_names)]
    seed.drop(columns=["normalized_cfpb_name", "normalized_canonical_name"]).to_parquet(parquet_path, index=False)
    review.to_csv(review_path, index=False)
    return {
        "distinct_cfpb_companies": int(companies.shape[0]),
        "seed_mappings": int(seed.shape[0]),
        "high_confidence_mappings": int((seed["confidence_tier"] == "HIGH CONFIDENCE").sum()),
        "medium_review_mappings": int(seed["confidence_tier"].str.startswith("MEDIUM").sum()),
        "review_candidates": int(review.shape[0]),
        "mapping_coverage_by_distinct_company": float(mapped_companies.shape[0] / max(companies.shape[0], 1)),
        "mapping_coverage_by_complaint_volume": float(mapped_companies["complaints"].sum() / max(companies["complaints"].sum(), 1)),
        "high_confidence_coverage_by_complaint_volume": float(high_companies["complaints"].sum() / max(companies["complaints"].sum(), 1)),
        "parquet_path": str(parquet_path),
        "review_path": str(review_path),
    }
