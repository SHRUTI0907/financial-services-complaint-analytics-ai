# Current State

This audit is based on the repository contents in `/Users/sarvesh/Documents/Project_resume`, not on the project description alone.

The project currently works as a CFPB complaint analytics app with a 250,000-record processed CFPB extract, Streamlit UI, data-quality profiling, complaint trend views, NMF topic outputs, a TF-IDF + Logistic Regression routing model report, value modeling, opportunity identification, NIST-informed governance docs, tests, and measured evidence artifacts.

# Verified Metrics

| Metric | Verified value | Source |
|---|---:|---|
| CFPB records processed | 250,000 | `data/processed/cfpb_complaints.parquet` |
| Date range | 2016-03-02 to 2026-08-09 | `data/processed/dataset_metadata.json` |
| Distinct companies | 1,383 | Parquet scan |
| Products | 12 | Parquet scan |
| Issues | 87 | Parquet scan |
| Public complaint narratives | 5,386 | Parquet scan |
| Topic rows | 12 | `data/artifacts/topic_registry.csv` |
| AI opportunity rows | 30 | `data/artifacts/ai_opportunities.csv` |
| Routing model training/eval records | 5,310 | `data/artifacts/model_report.json` |
| Routing product classes | 10 | `data/artifacts/model_report.json` |
| Baseline macro F1 | 0.052 | `data/artifacts/model_report.json` |
| Model macro F1 | 0.729 | `data/artifacts/model_report.json` |
| Existing RAG/evidence eval questions | 12 | `evaluation/rag_eval_set.csv` |

# Architecture

Current architecture:

```text
CFPB bulk/API ingestion
  -> normalized CFPB columns
  -> Parquet store
  -> quality profile
  -> trend/anomaly analytics
  -> NMF topics
  -> TF-IDF classifier
  -> opportunity/value/governance artifacts
  -> Streamlit app
```

Core folders:
- `src/ingestion`: CFPB, SEC, and BLS ingestion stubs/modules.
- `src/analytics`: complaint summaries, trends, anomaly scoring, benchmarking.
- `src/nlp`: text cleaning, topic discovery, retrieval, classification, basic RAG eval.
- `src/value`: scenario/value formulas and opportunity derivation.
- `src/governance`: NIST-informed control catalog.
- `src/reporting`: lineage and final evidence generation.
- `tests`: 17 unit tests.

# Strengths

- Uses real CFPB complaint data rather than synthetic complaints.
- Keeps large raw data out of the shareable release.
- Separates calculations from Streamlit UI.
- Has repeatable ingestion and artifact generation scripts.
- Includes useful tests around ingestion, analytics, NLP, value, governance, and retrieval eval.
- Clearly warns that CFPB complaint data is not a perfect sample of consumer experience.
- Has measured evidence docs instead of invented resume numbers.

# Technical Debt / Weaknesses

- Current evidence search is lexical TF-IDF only, not a proper hybrid RAG pipeline.
- Existing `rag_eval_set.csv` is small and evaluates retrieval lightly.
- No generated answer layer exists yet; the app only retrieves evidence rows.
- Topic registry contains topic summaries but not a saved per-complaint topic assignment artifact.
- SEC and BLS modules exist, but no verified SEC/BLS output artifacts are present in this release.
- Company matching exists as a seed CSV, but there is no layered entity-resolution workflow yet.
- Value model uses editable wage assumptions in the verified run, not fetched BLS benchmarks.
- Governance is useful but not yet tied to a formal AI-system inventory, risk register, and model cards.

# Missing Capabilities

| Capability | Current status |
|---|---|
| RAG | Partial. Evidence retrieval exists, but no cached hybrid dense + lexical index or answer composer. |
| LLM generation | Missing. No provider-agnostic answer layer exists in the verified app. |
| SEC integration | Module and seed map exist, but no verified SEC observations are present. |
| BLS integration | Module exists, but no verified BLS wage artifact is present. |
| FRED integration | Not implemented. |
| Entity resolution | Seed mapping exists; layered matching/review workflow is missing. |
| Evaluation | Basic tests and small retrieval eval exist; RAG-specific Recall@K/citation metrics need to be added. |
| Data lineage | Good for CFPB core; needs RAG and enrichment lineage. |
| Governance | Useful first pass; needs system inventory, model cards, risk register, and RAG-specific controls. |
| Value realization | Works as a scenario model; needs BLS-backed benchmark selection and Monte Carlo later. |

# Proposed Target Architecture

```text
Official data sources
  -> raw ingestion
  -> Parquet analytical layer
  -> entity resolution / enrichment
  -> deterministic analytics
  -> NLP / ML
  -> cached hybrid retrieval index
  -> analytics tool context
  -> provider-agnostic grounded answer layer
  -> Streamlit analyst app
```

This stays interview-friendly: Python modules, Parquet, scikit-learn, Streamlit, optional LLM provider. No microservices or unnecessary infrastructure.

# Upgrade Roadmap

## Phase 1 - RAG Foundations
- Build offline RAG index script.
- Preserve complaint metadata per indexed narrative.
- Add lexical TF-IDF retrieval.
- Add dense semantic retrieval using local LSA/SVD embeddings.
- Add hybrid score fusion and metadata filters.
- Add deterministic analytics tools for complaint metrics.
- Expand RAG evaluation to 40-60 questions.

## Phase 2 - LLM Answer Layer
- Add provider-agnostic answer service.
- Support no-key deterministic fallback.
- Add system prompt with strict citation and abstention rules.
- Add answer trace showing filters, analytics calls, retrieval scores, evidence, and citations.
- Upgrade Streamlit Evidence Search into AI Complaint Intelligence Analyst.

## Phase 3 - SEC Enrichment
- Run official SEC Companyfacts ingestion.
- Build auditable entity-resolution workflow.
- Add company-year financial scale observations and normalized complaint views.

## Phase 4 - BLS/FRED Enrichment
- Run official BLS wage benchmark ingestion.
- Add optional FRED macro context only where analytically useful.

## Phase 5 - Value-Realization Upgrades
- Add BLS-backed wage selection.
- Add Monte Carlo scenarios and distribution documentation.

## Phase 6 - Governance Upgrades
- Add AI system inventory, risk register, model cards, and RAG/LLM controls.

## Phase 7 - Evaluation
- Add RAG Recall@K, citation validity, unsupported-claim checks, and answer-quality audit artifacts.

## Phase 8 - UX Improvements
- Improve the Evidence Search page, traceability, and executive readouts.

## Phase 9 - Documentation / Resume Evidence
- Regenerate verified metrics and resume evidence only from actual outputs.

# Dependency Changes

Phase 1 and 2 can be implemented with the current stack:
- `scikit-learn`: TF-IDF, TruncatedSVD dense embeddings, cosine similarity.
- `joblib`: cached index artifacts.
- `pandas` / `pyarrow`: metadata and Parquet.

Optional future additions:
- `sentence-transformers`: stronger local embeddings if the environment can install model dependencies.
- `openai` / `anthropic`: SDK convenience. The current provider layer can also use HTTP calls, so SDKs are not required.

# Risks

- API limits: SEC/BLS/FRED and LLM calls need rate handling and clear user-agent/API-key configuration.
- Entity matching: CFPB company names may be subsidiaries, brands, servicers, or parent names.
- LLM cost: generated answers should be optional and disabled without keys.
- Hallucination: answers must use supplied metrics/evidence only and abstain when evidence is weak.
- Embedding scale: full CFPB narrative indexing may require batching and larger local storage.
- Financial comparability: assets/revenue are scale proxies, not customer count or market share.
- Privacy: CFPB narratives are public and redacted, but the app should still avoid exposing unnecessary text.
- Evaluation limitations: synthetic expected terms in evaluation questions are not the same as human relevance judgment.

# Recommended Next Step

Implement Phase 1 and Phase 2 now: cached hybrid retrieval, deterministic analytics tools, provider-agnostic answer generation, no-key fallback, expanded RAG evaluation, app UI upgrade, and updated evidence docs.
