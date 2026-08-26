# Redesign Plan

## Current Architecture
At inspection time, the repository contained only Git metadata and an initial `PROJECT_PLAN.md` describing a synthetic enterprise AI use-case platform. No application code, datasets, tests, or reusable calculations had been implemented.

## Components To Preserve
- The decision-support ambition.
- The emphasis on transparent financial modeling, scenario analysis, governance, and interview-defensible methodology.
- The Streamlit/Python orientation from the original brief.

## Components To Replace
- Synthetic AI use-case dataset.
- Fictional companies and fictional operational metrics.
- Prewritten executive findings.
- Any UI concept that starts from imaginary project portfolios rather than observed real-world operational problems.

## New Architecture
The redesigned project is a real-data complaint analysis product built on official public sources:

- CFPB Consumer Complaint Database as the operating-problem layer.
- SEC EDGAR/XBRL company facts as public-company scale metrics.
- BLS OEWS wage benchmarks as external labor-cost inputs.
- Optional FRED macro context for correlation-only time-series context.
- DuckDB and Parquet for scalable local analytics.
- scikit-learn NLP and anomaly models as a baseline that can be upgraded to embeddings/BERTopic when dependencies and compute are available.
- Streamlit for a clean analyst-facing app.

## Migration Steps
1. Remove the synthetic-data plan from the product path.
2. Create source registry and reproducible ingestion scripts.
3. Build CFPB schema normalization, Parquet caching, DuckDB views, and metadata capture.
4. Add data-quality profiling and validation checks.
5. Add SEC entity-resolution seed map and official EDGAR fetcher.
6. Add BLS wage benchmark ingestion.
7. Build complaint analytics, time-series anomaly detection, and company benchmarking.
8. Add narrative NLP: cleaning, topic discovery, routing model, retrieval, and evaluation hooks.
9. Derive AI opportunities from observed complaint patterns.
10. Model value realization from observed complaint volume, BLS wage benchmarks, and user-controlled assumptions.
11. Add NIST-aligned governance requirements for implemented AI systems.
12. Build the Streamlit app and methodology/data-lineage documentation.

## Data Policy
Large raw datasets are not committed. The repository contains scripts to rebuild local analytical stores from official sources. Small test fixtures may be used only for automated tests and are not presented as analytical findings.

## Redesign Decision
The synthetic prototype is not archived because no synthetic implementation existed. If future experiments require synthetic mocks, they should live outside the core analytical product and be clearly marked as prototype-only.
