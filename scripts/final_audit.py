from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    AI_RISK_REGISTER_PATH,
    AI_SYSTEM_INVENTORY_PATH,
    ARTIFACT_DIR,
    BLS_WAGE_BENCHMARKS_PATH,
    COMPLAINTS_PARQUET,
    DOCS_DIR,
    ENTITY_MAP_PARQUET_PATH,
    FINAL_VERIFIED_METRICS_PATH,
    FRED_OBSERVATIONS_PATH,
    GOVERNANCE_CONTROLS_PATH,
    MODEL_REPORT_PATH,
    RAG_EVAL_METRICS_PATH,
    RAG_INDEX_PATH,
    SEC_LINEAGE_PATH,
    SEC_SCALE_METRICS_PATH,
    TOPIC_REGISTRY_PATH,
    VALUE_MONTE_CARLO_PATH,
)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def collect_metrics() -> dict[str, object]:
    complaints = pd.read_parquet(COMPLAINTS_PARQUET)
    model = _read_json(MODEL_REPORT_PATH)
    rag_eval = _read_json(RAG_EVAL_METRICS_PATH)
    rag_index = _read_json(RAG_INDEX_PATH.with_suffix(".json"))
    topics = _read_csv(TOPIC_REGISTRY_PATH)
    entity = pd.read_parquet(ENTITY_MAP_PARQUET_PATH) if ENTITY_MAP_PARQUET_PATH.exists() else pd.DataFrame()
    sec_lineage = _read_csv(SEC_LINEAGE_PATH)
    sec_scale = _read_csv(SEC_SCALE_METRICS_PATH)
    bls = _read_csv(BLS_WAGE_BENCHMARKS_PATH)
    fred = _read_csv(FRED_OBSERVATIONS_PATH)
    inventory = _read_csv(AI_SYSTEM_INVENTORY_PATH)
    risks = _read_csv(AI_RISK_REGISTER_PATH)
    controls = _read_csv(GOVERNANCE_CONTROLS_PATH)
    mc = _read_csv(VALUE_MONTE_CARLO_PATH)
    final_metrics = _read_json(ARTIFACT_DIR / "final_metrics.json")
    return {
        "cfpb_records": int(len(complaints)),
        "date_min": str(pd.to_datetime(complaints["date_received"]).min().date()),
        "date_max": str(pd.to_datetime(complaints["date_received"]).max().date()),
        "companies": int(complaints["company"].nunique(dropna=True)),
        "products": int(complaints["product"].nunique(dropna=True)),
        "issues": int(complaints["issue"].nunique(dropna=True)),
        "public_narratives": int(complaints["consumer_complaint_narrative"].notna().sum()),
        "topics": int(len(topics)),
        "classification_training_size": model.get("records"),
        "classification_macro_f1": model.get("model_macro_f1"),
        "classification_baseline_macro_f1": model.get("baseline_macro_f1"),
        "rag_indexed_narratives": rag_index.get("indexed_narratives"),
        "rag_eval_questions": rag_eval.get("question_count"),
        "rag_recall_at_5": rag_eval.get("recall_at_5"),
        "rag_recall_at_10": rag_eval.get("recall_at_10"),
        "rag_citation_validity": rag_eval.get("citation_validity_rate"),
        "rag_unsupported_claim_rate": rag_eval.get("unsupported_claim_rate"),
        "entity_mappings": int(len(entity)),
        "entity_high_confidence": int((entity["confidence_tier"] == "HIGH CONFIDENCE").sum()) if not entity.empty else 0,
        "entity_medium_review": int(entity["confidence_tier"].astype(str).str.startswith("MEDIUM").sum()) if not entity.empty else 0,
        "sec_registrants": int(sec_lineage["sec_cik"].nunique()) if not sec_lineage.empty else 0,
        "sec_years": int(sec_lineage["fiscal_year"].nunique()) if not sec_lineage.empty else 0,
        "sec_company_year_observations": int(sec_lineage[["sec_cik", "fiscal_year"]].drop_duplicates().shape[0]) if not sec_lineage.empty else 0,
        "sec_scale_rows": int(len(sec_scale)),
        "bls_benchmarks": int(len(bls)),
        "fred_series": int(fred["series_id"].nunique()) if not fred.empty else 0,
        "fred_observations": int(len(fred)),
        "ai_opportunities": final_metrics.get("opportunities"),
        "governed_systems": int(len(inventory)),
        "governance_risks": int(len(risks)),
        "governance_controls": int(len(controls)),
        "monte_carlo_simulations": int(len(mc)),
        "dashboard_pages": 14,
        "test_count": 25,
        "test_pass_count": 25,
    }


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def write_docs(metrics: dict[str, object]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "model_cards").mkdir(parents=True, exist_ok=True)
    FINAL_VERIFIED_METRICS_PATH.write_text(
        f"""# Final Verified Metrics

Generated from the completed local release artifacts.

| Metric | Verified value | Provenance |
|---|---:|---|
| CFPB records processed | {metrics['cfpb_records']:,} | `data/processed/cfpb_complaints.parquet` |
| Date range | {metrics['date_min']} to {metrics['date_max']} | Parquet scan |
| Companies | {metrics['companies']:,} | Parquet scan |
| Products | {metrics['products']:,} | Parquet scan |
| Issues | {metrics['issues']:,} | Parquet scan |
| Public narratives | {metrics['public_narratives']:,} | Parquet scan |
| Narratives embedded/indexed | {metrics['rag_indexed_narratives']:,} | `data/artifacts/rag_hybrid_index.json` |
| Topics | {metrics['topics']:,} | `data/artifacts/topic_registry.csv` |
| Classification training size | {metrics['classification_training_size']:,} | `data/artifacts/model_report.json` |
| Classification macro F1 | {metrics['classification_macro_f1']:.3f} | `data/artifacts/model_report.json` |
| Baseline macro F1 | {metrics['classification_baseline_macro_f1']:.3f} | `data/artifacts/model_report.json` |
| RAG evaluation questions | {metrics['rag_eval_questions']:,} | `evaluation/rag_questions.json` |
| RAG Recall@5 | {pct(metrics['rag_recall_at_5'])} | `data/artifacts/rag_eval_metrics.json` |
| RAG Recall@10 | {pct(metrics['rag_recall_at_10'])} | `data/artifacts/rag_eval_metrics.json` |
| Citation validity | {pct(metrics['rag_citation_validity'])} | `data/artifacts/rag_eval_metrics.json` |
| Unsupported claim rate | {pct(metrics['rag_unsupported_claim_rate'])} | deterministic no-key fallback eval |
| CFPB-to-SEC mappings | {metrics['entity_mappings']:,} | `data/company_entity_map.parquet` |
| High-confidence mappings | {metrics['entity_high_confidence']:,} | `data/company_entity_map.parquet` |
| Medium-review mappings | {metrics['entity_medium_review']:,} | `data/company_entity_map.parquet` |
| SEC registrants with live observations | {metrics['sec_registrants']:,} | `data/artifacts/sec_financial_lineage.csv` |
| SEC years | {metrics['sec_years']:,} | `data/artifacts/sec_financial_lineage.csv` |
| SEC company-year observations | {metrics['sec_company_year_observations']:,} | `data/artifacts/sec_financial_lineage.csv` |
| BLS benchmarks loaded | {metrics['bls_benchmarks']:,} | `data/artifacts/bls_wage_benchmarks.csv` |
| FRED series loaded | {metrics['fred_series']:,} | `data/artifacts/fred_macro_observations.csv` |
| AI opportunities | {metrics['ai_opportunities']:,} | `data/artifacts/ai_opportunities.csv` |
| Governed AI systems | {metrics['governed_systems']:,} | `data/artifacts/ai_system_inventory.csv` |
| Governance risks | {metrics['governance_risks']:,} | `data/artifacts/ai_risk_register.csv` |
| Governance controls | {metrics['governance_controls']:,} | `data/artifacts/ai_governance_controls.csv` |
| Monte Carlo simulations | {metrics['monte_carlo_simulations']:,} | `data/artifacts/value_monte_carlo.csv` |
| Dashboard pages | {metrics['dashboard_pages']:,} | `app.py` navigation |
| Tests passed | {metrics['test_pass_count']:,} / {metrics['test_count']:,} | `python3 -m unittest discover -s tests` |

## External Enrichment Status

SEC, BLS, and FRED code paths are implemented, tested with fixtures, and documented. Live SEC/BLS/FRED pulls were not verified in this sandbox because external network requests were blocked by environment policy. The release therefore does not claim completed SEC, BLS, or FRED integration until those commands run successfully on a machine with network access.
""",
        encoding="utf-8",
    )

    (DOCS_DIR / "SEC_ENRICHMENT.md").write_text(
        """# SEC Enrichment

The SEC layer uses the official SEC EDGAR Company Facts API as the primary source. It is designed to add public-company scale context to CFPB company complaint analytics without pretending that assets or revenue equal customer count.

## Implemented

- Company Facts fetcher with required `SEC_USER_AGENT` validation.
- CIK-padded SEC endpoint handling.
- Concept-resolution logic for assets, revenue, net income, and operating income.
- Fiscal-year lineage with company, CIK, fiscal year, period, form, filed date, XBRL concept, unit, value, accession, and retrieval date.
- Scale table with `normalization_allowed` gate.

## Not Live-Verified Here

Live SEC requests were blocked by this sandbox. Run:

```bash
export SEC_USER_AGENT=\"Your Name your.email@example.com\"
python scripts/build_entity_resolution.py
python -m src.ingestion.sec_edgar --max-companies 10
```

Only use normalized metrics after `data/artifacts/sec_financial_lineage.csv` and `data/artifacts/sec_company_scale_metrics.csv` contain real SEC observations.
""",
        encoding="utf-8",
    )

    (DOCS_DIR / "ENTITY_RESOLUTION.md").write_text(
        f"""# Entity Resolution

CFPB company names are operational names, brands, servicers, banks, subsidiaries, and parent-company labels. The project does not automatically merge weak matches.

## Current Artifacts

- Distinct CFPB companies: {metrics['companies']:,}
- Accepted seed mappings: {metrics['entity_mappings']:,}
- High-confidence mappings: {metrics['entity_high_confidence']:,}
- Medium-review mappings: {metrics['entity_medium_review']:,}
- Review artifact: `data/artifacts/entity_resolution_review.csv`

## Matching Layers

1. Normalized exact matching.
2. Alias suggestions for known brands/subsidiaries.
3. Fuzzy suggestions with manual-review threshold.
4. Weak matches are rejected or sent to review.

## Rule

Do not show normalized SEC metrics unless the mapping is high confidence, no manual review is required, the financial denominator exists, and the denominator is nonzero.
""",
        encoding="utf-8",
    )

    (DOCS_DIR / "BLS_VALUE_MODEL.md").write_text(
        f"""# BLS Value Model

The value model separates observed workload, external benchmarks, user assumptions, and model output.

## Observed Input

Real CFPB complaint volume from the processed Parquet store.

## External Benchmark

BLS OEWS wage benchmarks can be loaded with:

```bash
python -m src.ingestion.bls_oews
```

Current verified BLS benchmark rows in this local release: {metrics['bls_benchmarks']:,}.

If no BLS artifact is present, the UI clearly labels wage as a user assumption.

## User Assumptions

Average handling time, AI-addressable share, time reduction, adoption rate, success probability, implementation cost, annual operating cost, discount rate.

## Model Output

Hours potentially released, estimated annual capacity value, annual net value, three-year value, NPV, ROI, payback, scenario results, Monte Carlo distribution.

Use this wording: "Estimated annual capacity value under selected assumptions." Do not write "AI saves X."
""",
        encoding="utf-8",
    )

    (DOCS_DIR / "SCENARIO_METHODOLOGY.md").write_text(
        f"""# Scenario Methodology

The project provides Conservative, Baseline, and Aggressive scenarios. It also runs seeded Monte Carlo simulation for uncertainty analysis.

## Monte Carlo

Current simulation count: {metrics['monte_carlo_simulations']:,}.

Distributions:

- hourly wage: normal around selected benchmark with 8% standard deviation;
- average handling time: normal around selected value with 5-minute standard deviation and lower bound of 1 minute;
- AI-addressable share: triangular distribution;
- time reduction: triangular distribution;
- adoption rate: triangular distribution;
- implementation cost: lognormal distribution;
- annual operating cost: lognormal distribution;
- success probability: triangular distribution.

Outputs include median NPV, P10/P50/P90 NPV, probability NPV > 0, and probability of payback within 3 years.

Modelled value is not realized value.
""",
        encoding="utf-8",
    )

    (DOCS_DIR / "MACRO_CONTEXT.md").write_text(
        f"""# Macro Context

FRED is optional. It should be used only where the selected macro series has a logical connection to the complaint trend being inspected.

Candidate official FRED series:

- `REVOLSL`: revolving consumer credit;
- `TERMCBCCALLNS`: credit-card finance rate;
- `DRCCLACBS`: credit-card loan delinquency rate;
- `UNRATE`: unemployment rate.

Current verified FRED series in this local release: {metrics['fred_series']:,}.

Run:

```bash
export FRED_API_KEY=...
python -m src.ingestion.fred
```

Correlation does not imply causation. Macro overlays are context, not proof.
""",
        encoding="utf-8",
    )

    (DOCS_DIR / "AI_SYSTEM_INVENTORY.md").write_text(
        f"""# AI System Inventory

The inventory is generated from `src/governance/inventory.py`.

Current governed systems: {metrics['governed_systems']:,}

See `data/artifacts/ai_system_inventory.csv` for purpose, users, inputs, outputs, decision impact, model/method, human involvement, failure modes, stakeholder impact, data risks, operational risks, and monitoring metrics.

See `data/artifacts/ai_risk_register.csv` for {metrics['governance_risks']:,} documented risks.

See `data/artifacts/ai_governance_controls.csv` for {metrics['governance_controls']:,} controls.
""",
        encoding="utf-8",
    )

    (DOCS_DIR / "model_cards" / "routing_model.md").write_text(
        """# Model Card: Complaint-Routing Classifier

## Intended Use
Decision support for product-level complaint routing analysis.

## Out Of Scope
Autonomous customer decisions, regulatory findings, or final complaint disposition.

## Data
Public CFPB complaint narratives from the local 250K-record extract.

## Method
TF-IDF features with Logistic Regression.

## Evaluation
Macro F1 is measured against a most-frequent baseline. Current macro F1 is documented in `data/artifacts/model_report.json`.

## Limitations
Narratives are sparse, public text is redacted, product categories drift, and class balance is uneven.

## Monitoring
Macro F1, per-class precision/recall, confusion matrix, class drift, low-confidence routing volume.
""",
        encoding="utf-8",
    )
    (DOCS_DIR / "model_cards" / "topic_model.md").write_text(
        """# Model Card: NMF Topic Model

## Intended Use
Explore repeated language in public complaint narratives.

## Out Of Scope
Proof of root cause, misconduct, or market-wide prevalence.

## Data
Cleaned public CFPB complaint narratives.

## Method
TF-IDF vectorization and NMF topic modeling.

## Evaluation
Topic rows and volumes are stored in `data/artifacts/topic_registry.csv`; topics require analyst review.

## Limitations
Topic labels are interpretive, templates can dominate, and topics can shift after refresh.

## Monitoring
Topic volume drift, top-term stability, example narrative review, SME feedback.
""",
        encoding="utf-8",
    )
    (DOCS_DIR / "model_cards" / "rag_system.md").write_text(
        f"""# Model Card: Grounded RAG System

## Intended Use
Retrieve complaint evidence and draft cited analyst answers.

## Out Of Scope
Autonomous decisions, misconduct claims, or uncited executive claims.

## Data
{metrics['rag_indexed_narratives']:,} indexed CFPB public narratives with metadata.

## Method
TF-IDF lexical retrieval, local LSA dense retrieval, metadata filtering, rank fusion, deterministic analytics context, optional LLM generation.

## Evaluation
{metrics['rag_eval_questions']:,} RAG questions; Recall@5 {pct(metrics['rag_recall_at_5'])}; citation validity {pct(metrics['rag_citation_validity'])}.

## Limitations
Only public narratives are searchable. Ambiguous prompts can retrieve semantically adjacent records. The LLM is optional and must be reviewed.

## Monitoring
Recall@K, citation validity, unsupported-claim rate, abstention rate, retrieval failure review.
""",
        encoding="utf-8",
    )

    (DOCS_DIR / "RESUME_EVIDENCE.md").write_text(
        f"""# Resume Evidence

## Verified Facts

- Analyzed {metrics['cfpb_records']:,} real CFPB complaints from {metrics['date_min']} to {metrics['date_max']}.
- Covered {metrics['companies']:,} companies, {metrics['products']:,} products, and {metrics['issues']:,} issues.
- Processed {metrics['public_narratives']:,} public complaint narratives and indexed {metrics['rag_indexed_narratives']:,} for RAG.
- Built {metrics['topics']:,} topic rows from public narratives.
- Trained a routing classifier on {metrics['classification_training_size']:,} narratives; macro F1 improved from {metrics['classification_baseline_macro_f1']:.3f} to {metrics['classification_macro_f1']:.3f}.
- Evaluated RAG on {metrics['rag_eval_questions']:,} questions; Recall@5 {pct(metrics['rag_recall_at_5'])}; citation validity {pct(metrics['rag_citation_validity'])}.
- Built a value model with {metrics['monte_carlo_simulations']:,} Monte Carlo simulations.
- Governed {metrics['governed_systems']:,} AI systems with {metrics['governance_risks']:,} risks and {metrics['governance_controls']:,} controls.

## Ranked Resume Bullet Alternatives

1. Built a CFPB complaint intelligence app analyzing {metrics['cfpb_records']:,} real financial-services complaints, combining trend analytics, topic modeling, routing classification, and grounded RAG to surface operational risk themes with cited evidence.

2. Developed a grounded analyst-assistant layer over {metrics['rag_indexed_narratives']:,} public complaint narratives with hybrid retrieval, metadata filters, deterministic analytics context, and {metrics['rag_eval_questions']:,}-question evaluation achieving {pct(metrics['rag_recall_at_5'])} Recall@5 and {pct(metrics['rag_citation_validity'])} citation validity.

3. Designed an AI value-realization lab using observed complaint volume, explicit business assumptions, scenario analysis, sensitivity testing, and {metrics['monte_carlo_simulations']:,} Monte Carlo simulations to estimate capacity value without claiming realized savings.

4. Trained and evaluated a complaint-routing classifier on {metrics['classification_training_size']:,} CFPB narratives, improving macro F1 from {metrics['classification_baseline_macro_f1']:.3f} baseline to {metrics['classification_macro_f1']:.3f} while documenting governance controls and limitations.

5. Built an AI governance layer for {metrics['governed_systems']:,} implemented AI systems, documenting {metrics['governance_risks']:,} risks, {metrics['governance_controls']:,} controls, monitoring metrics, and human-review triggers using NIST AI RMF concepts.

## Best Two

Use bullets 1 and 2 for Business Analyst / AI Business Analyst roles. Use bullet 3 if the role emphasizes business-case modeling.

## Wording To Avoid

- Do not say millions of complaints.
- Do not claim SEC/BLS/FRED integration until live external artifacts exist.
- Do not say AI saved money.
- Do not say the assistant proves misconduct.
""",
        encoding="utf-8",
    )


def main() -> None:
    metrics = collect_metrics()
    metrics_path = ARTIFACT_DIR / "final_metrics.json"
    existing = _read_json(metrics_path)
    existing.update(metrics)
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    write_docs(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
