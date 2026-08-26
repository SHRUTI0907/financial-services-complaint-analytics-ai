from __future__ import annotations

import argparse

import pandas as pd

from src.config import AI_OPPORTUNITIES_PATH, QUALITY_PROFILE_PATH, TOPIC_REGISTRY_PATH, ensure_directories
from src.ingestion.cfpb import ingest_cfpb_bulk
from src.nlp.modeling import train_issue_classifier
from src.nlp.topics import discover_topics
from src.quality.profile import build_quality_profile, validate_complaints_schema
from src.analytics.anomaly import emerging_issue_score
from src.value.opportunities import derive_ai_opportunities


def run_local_pipeline(max_records: int | None = None, run_nlp: bool = True) -> dict:
    ensure_directories()
    metadata = ingest_cfpb_bulk(max_records=max_records)
    df = pd.read_parquet(metadata["output_path"])
    errors = validate_complaints_schema(df)
    if errors:
        raise ValueError("; ".join(errors))
    build_quality_profile(df).to_csv(QUALITY_PROFILE_PATH, index=False)
    emerging = emerging_issue_score(df, "issue")
    opportunities = derive_ai_opportunities(df, emerging=emerging)
    opportunities.to_csv(AI_OPPORTUNITIES_PATH, index=False)
    if run_nlp:
        topics, _ = discover_topics(df)
        topics.to_csv(TOPIC_REGISTRY_PATH, index=False)
        train_issue_classifier(df)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CFPB-centered local analytical pipeline.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional development cap. Omit for full official dataset.")
    parser.add_argument("--skip-nlp", action="store_true")
    args = parser.parse_args()
    print(run_local_pipeline(max_records=args.max_records, run_nlp=not args.skip_nlp))


if __name__ == "__main__":
    main()
