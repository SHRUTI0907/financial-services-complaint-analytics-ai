# CFPB Complaint Intelligence + AI Governance Lab

This is a portfolio project for business analytics, AI transformation, and decision-support roles. It turns real CFPB consumer complaint records into an analyst workspace for complaint trends, operational friction, topic discovery, routing-model evaluation, grounded RAG, value modeling, and AI governance.

It is not a fake consulting mockup. The current local release is built on a capped official CFPB extract of 250,000 real complaint records.

## What This Solves

Financial-services teams need to know where consumers are seeing repeated friction, which complaint themes are rising, where AI support might help, and what controls would be required before using AI in a real workflow.

This project answers those questions with:

- observed complaint data first;
- deterministic metrics before generated text;
- cited complaint evidence for analyst answers;
- scenario/value modeling with visible assumptions;
- governance tied to the actual AI systems in the repo.

## Real-World Data

Verified local release:

| Metric | Value |
|---|---:|
| CFPB records | 250,000 |
| Date range | 2016-03-02 to 2026-08-09 |
| Companies | 1,383 |
| Products | 12 |
| Issues | 87 |
| Public narratives | 5,386 |
| RAG-indexed narratives | 5,343 |

SEC, BLS, and FRED pipelines are implemented but were not live-verified in this sandbox because external network requests were blocked. The app does not claim completed SEC/BLS/FRED enrichment until those artifacts exist.

## Key Capabilities

- CFPB ingestion into Parquet.
- Data-quality and lineage artifacts.
- Complaint trend, product, issue, company, response, and state analysis.
- Emerging-risk monitor using transparent z-score logic.
- NMF topic modeling over public narratives.
- Complaint-routing classifier using TF-IDF + Logistic Regression.
- Grounded RAG over public CFPB narratives with metadata filters and complaint citations.
- Optional LLM answer layer with deterministic no-key fallback.
- Scenario value model with NPV, ROI, payback, sensitivity, and Monte Carlo.
- AI governance inventory, risk register, controls, and model cards.

## Measured Results

| Metric | Value |
|---|---:|
| Routing baseline macro F1 | 0.052 |
| Routing model macro F1 | 0.729 |
| RAG evaluation questions | 45 |
| RAG Recall@5 | 97.8% |
| RAG Recall@10 | 97.8% |
| RAG citation validity | 100.0% |
| Monte Carlo simulations | 5,000 |
| Governed AI systems | 6 |
| Governance risks | 14 |
| Governance controls | 12 |
| Tests | 25 passing |

Full metric provenance is in `docs/FINAL_VERIFIED_METRICS.md`.

## Architecture

```text
Official CFPB data
  -> ingestion and normalized Parquet
  -> quality checks and lineage
  -> deterministic analytics
  -> NLP topics and routing model
  -> hybrid RAG index
  -> grounded answer layer
  -> value modeling and governance
  -> Streamlit executive app
```

Optional future live enrichments:

```text
SEC Company Facts -> audited entity map -> scale-normalized complaint view
BLS OEWS -> wage benchmark -> value model
FRED -> macro context overlay
```

## How To Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline --max-records 250000
python scripts/build_rag_index.py
python scripts/evaluate_rag.py
python scripts/build_entity_resolution.py
python scripts/build_governance_artifacts.py
python scripts/run_value_analysis.py
python scripts/final_audit.py
python3 -m streamlit run app.py
```

## Optional External Enrichment

SEC:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
python -m src.ingestion.sec_edgar --max-companies 10
```

BLS:

```bash
python -m src.ingestion.bls_oews
```

FRED:

```bash
export FRED_API_KEY=...
python -m src.ingestion.fred
```

## Methodology

The project separates:

- observed data: CFPB complaint records;
- external benchmarks: SEC/BLS/FRED only after live artifacts exist;
- user assumptions: handling time, adoption, time reduction, costs, success probability;
- model outputs: estimated capacity value, NPV, ROI, payback, and probability ranges.

The LLM does not calculate business metrics. It can only draft from retrieved evidence and deterministic analytics context.

## Limitations

- CFPB complaints are not a full sample of all consumer experiences.
- Public narratives are sparse and redacted.
- SEC assets/revenue are imperfect scale proxies and are not customer count.
- BLS wages are external benchmarks, not company payroll data.
- Scenario values are not realized savings.
- Macro correlations, if FRED is enabled, do not imply causation.
