# Financial Services AI Operations & Governance Intelligence Platform

## New Direction
This project is built around real U.S. public financial-services data, not a synthetic AI use-case dataset. The product connects complaint analysis, entity normalization, financial-scale benchmarking, labor-cost benchmarks, NLP, anomaly detection, value modeling, and NIST-aligned AI governance.

## Core Product Story
Real consumer complaint data -> operational pain-point discovery -> emerging issue detection -> company and product benchmarking -> AI opportunity discovery -> transparent value modeling -> governed AI implementation recommendations.

## Data Sources
- CFPB Consumer Complaint Database: primary operating dataset, ingested from the official CFPB bulk download or API.
- SEC EDGAR/XBRL: public-company financial scale metrics for defensibly matched institutions.
- BLS OEWS: external labor-cost benchmarks for operational roles.
- FRED: optional macro context, enabled only when analytically useful and configured.

## Architecture
- `app.py`: Streamlit analyst app.
- `src/ingestion/`: official-source ingestion and refresh scripts.
- `src/storage/`: DuckDB/Parquet access layer.
- `src/quality/`: data profiling and validation.
- `src/analytics/`: complaint metrics, time-series intelligence, benchmarking, and anomaly detection.
- `src/nlp/`: narrative cleaning, topic discovery, routing model, retrieval, and RAG evaluation.
- `src/value/`: AI opportunity generation and value realization scenarios.
- `src/governance/`: NIST-aligned risk/control rules.
- `src/reporting/`: data lineage and benchmark report generation.
- `tests/`: unit tests with small local fixtures.

## Implementation Principles
- Use only real public data in the analytical product.
- Do not force entity matches or normalize complaints without clear denominator limitations.
- Keep all assumptions visible and editable.
- Separate observed evidence, external benchmarks, user assumptions, and model outputs.
- Cache expensive preprocessing so the app does not recompute topics/models on startup.
- Do not claim causation, regulatory compliance, or realized savings.

## Current Status
The repository was empty except for an initial synthetic-project plan. That plan has been replaced by this real-data redesign plan, and the synthetic approach has been removed from the implementation direction.
