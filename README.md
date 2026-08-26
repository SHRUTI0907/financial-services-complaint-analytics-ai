# Financial Services Complaint Analytics & AI Decision Support Platform

An interactive analytics application for exploring Consumer Financial Protection Bureau complaint data across companies, products, issues, narrative themes, and model outputs.

## What It Does

- Analyzes 250,000 real CFPB complaints across 1,383 companies.
- Evaluates 12 financial products and 87 issue categories.
- Surfaces trend and anomaly signals for complaint patterns.
- Uses NMF topic modeling and grounded RAG over public complaint narratives.
- Evaluates a 10-class complaint-routing model with 0.729 macro F1.

## Data

Source: [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)

Verified local release:

| Metric | Value |
|---|---:|
| CFPB complaints | 250,000 |
| Companies | 1,383 |
| Products | 12 |
| Issues | 87 |
| Public narratives | 5,386 |
| RAG-indexed narratives | 5,343 |
| Date range | 2016-03-02 to 2026-08-09 |

## App Sections

| Section | Purpose |
|---|---|
| Overview | Complaint KPIs and volume trends |
| Complaint Patterns | Product and issue analysis with anomaly signals |
| Company View | Company-level complaint volume and product mix |
| Narrative Search | NMF topics and grounded narrative evidence |
| Model Performance | Routing model and retrieval evaluation |

## Measured Results

| Metric | Value |
|---|---:|
| Routing model classes | 10 |
| Routing model macro F1 | 0.729 |
| Baseline macro F1 | 0.052 |
| RAG evaluation questions | 45 |
| RAG Recall@5 | 97.8% |
| RAG citation validity | 100.0% |
| Tests | 25 passing |

Detailed metric provenance is in `docs/FINAL_VERIFIED_METRICS.md`.

## Architecture

```text
Official CFPB data
  -> Parquet analytics dataset
  -> Python/Pandas transformations
  -> trend and anomaly analysis
  -> NMF topic modeling
  -> TF-IDF routing model
  -> hybrid RAG index
  -> Streamlit decision-support app
```

## Run Locally

```bash
cd financial-services-complaint-analytics-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app.py
```

The processed CFPB release artifacts are included so the app can run after installing requirements.

## Streamlit Cloud

Use these deploy settings:

| Setting | Value |
|---|---|
| Repository | `SHRUTI0907/financial-services-complaint-analytics-ai` |
| Branch | `main` |
| Main file path | `app.py` |

## Limitations

- CFPB complaints are consumer-submitted records, not a complete measure of all customer experience.
- Public narratives are sparse and may be redacted.
- Company comparisons are raw complaint counts, not market-share-adjusted rates.
- RAG answers are grounded in retrieved complaint narratives and should not be treated as legal or regulatory conclusions.
