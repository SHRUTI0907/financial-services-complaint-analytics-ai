from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = DATA_DIR / "artifacts"
CONFIG_DIR = ROOT_DIR / "config"
DOCS_DIR = ROOT_DIR / "docs"

COMPLAINTS_PARQUET = PROCESSED_DIR / "cfpb_complaints.parquet"
COMPLAINTS_SAMPLE_PARQUET = PROCESSED_DIR / "cfpb_complaints_sample.parquet"
DUCKDB_PATH = PROCESSED_DIR / "fs_ai_intelligence.duckdb"
METADATA_PATH = PROCESSED_DIR / "dataset_metadata.json"
QUALITY_PROFILE_PATH = ARTIFACT_DIR / "data_quality_profile.csv"
TOPIC_REGISTRY_PATH = ARTIFACT_DIR / "topic_registry.csv"
AI_OPPORTUNITIES_PATH = ARTIFACT_DIR / "ai_opportunities.csv"
MODEL_REPORT_PATH = ARTIFACT_DIR / "model_report.json"
RAG_EVAL_PATH = ARTIFACT_DIR / "rag_eval_report.csv"
RAG_INDEX_PATH = ARTIFACT_DIR / "rag_hybrid_index.joblib"
RAG_METADATA_PATH = ARTIFACT_DIR / "rag_index_metadata.parquet"
RAG_EVAL_JSON_PATH = ROOT_DIR / "evaluation" / "rag_questions.json"
RAG_EVAL_METRICS_PATH = ARTIFACT_DIR / "rag_eval_metrics.json"
ENTITY_MAP_CSV_PATH = DATA_DIR / "company_entity_map.csv"
ENTITY_MAP_PARQUET_PATH = DATA_DIR / "company_entity_map.parquet"
ENTITY_REVIEW_PATH = ARTIFACT_DIR / "entity_resolution_review.csv"
SEC_COMPANY_FACTS_DIR = RAW_DIR / "sec_companyfacts"
SEC_LINEAGE_PATH = ARTIFACT_DIR / "sec_financial_lineage.csv"
SEC_SCALE_METRICS_PATH = ARTIFACT_DIR / "sec_company_scale_metrics.csv"
BLS_WAGE_BENCHMARKS_PATH = ARTIFACT_DIR / "bls_wage_benchmarks.csv"
BLS_STATUS_PATH = ARTIFACT_DIR / "bls_ingestion_status.json"
FRED_OBSERVATIONS_PATH = ARTIFACT_DIR / "fred_macro_observations.csv"
FRED_ANALYSIS_PATH = ARTIFACT_DIR / "fred_macro_analysis.csv"
VALUE_MONTE_CARLO_PATH = ARTIFACT_DIR / "value_monte_carlo.csv"
VALUE_SCENARIO_PATH = ARTIFACT_DIR / "value_scenarios.csv"
AI_SYSTEM_INVENTORY_PATH = ARTIFACT_DIR / "ai_system_inventory.csv"
AI_RISK_REGISTER_PATH = ARTIFACT_DIR / "ai_risk_register.csv"
GOVERNANCE_CONTROLS_PATH = ARTIFACT_DIR / "ai_governance_controls.csv"
FINAL_VERIFIED_METRICS_PATH = DOCS_DIR / "FINAL_VERIFIED_METRICS.md"


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, ARTIFACT_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    SEC_COMPANY_FACTS_DIR.mkdir(parents=True, exist_ok=True)
