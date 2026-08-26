# Final Verified Metrics

Generated from the completed local release artifacts.

| Metric | Verified value | Provenance |
|---|---:|---|
| CFPB records processed | 250,000 | `data/processed/cfpb_complaints.parquet` |
| Date range | 2016-03-02 to 2026-08-09 | Parquet scan |
| Companies | 1,383 | Parquet scan |
| Products | 12 | Parquet scan |
| Issues | 87 | Parquet scan |
| Public narratives | 5,386 | Parquet scan |
| Narratives embedded/indexed | 5,343 | `data/artifacts/rag_hybrid_index.json` |
| Topics | 12 | `data/artifacts/topic_registry.csv` |
| Classification training size | 5,310 | `data/artifacts/model_report.json` |
| Classification macro F1 | 0.729 | `data/artifacts/model_report.json` |
| Baseline macro F1 | 0.052 | `data/artifacts/model_report.json` |
| RAG evaluation questions | 45 | `evaluation/rag_questions.json` |
| RAG Recall@5 | 97.8% | `data/artifacts/rag_eval_metrics.json` |
| RAG Recall@10 | 97.8% | `data/artifacts/rag_eval_metrics.json` |
| Citation validity | 100.0% | `data/artifacts/rag_eval_metrics.json` |
| Unsupported claim rate | 0.0% | deterministic no-key fallback eval |
| CFPB-to-SEC mappings | 15 | `data/company_entity_map.parquet` |
| High-confidence mappings | 10 | `data/company_entity_map.parquet` |
| Medium-review mappings | 5 | `data/company_entity_map.parquet` |
| SEC registrants with live observations | 0 | `data/artifacts/sec_financial_lineage.csv` |
| SEC years | 0 | `data/artifacts/sec_financial_lineage.csv` |
| SEC company-year observations | 0 | `data/artifacts/sec_financial_lineage.csv` |
| BLS benchmarks loaded | 0 | `data/artifacts/bls_wage_benchmarks.csv` |
| FRED series loaded | 0 | `data/artifacts/fred_macro_observations.csv` |
| AI opportunities | 30 | `data/artifacts/ai_opportunities.csv` |
| Governed AI systems | 6 | `data/artifacts/ai_system_inventory.csv` |
| Governance risks | 14 | `data/artifacts/ai_risk_register.csv` |
| Governance controls | 12 | `data/artifacts/ai_governance_controls.csv` |
| Monte Carlo simulations | 5,000 | `data/artifacts/value_monte_carlo.csv` |
| Dashboard pages | 14 | `app.py` navigation |
| Tests passed | 25 / 25 | `python3 -m unittest discover -s tests` |

## External Enrichment Status

SEC, BLS, and FRED code paths are implemented, tested with fixtures, and documented. Live SEC/BLS/FRED pulls were not verified in this sandbox because external network requests were blocked by environment policy. The release therefore does not claim completed SEC, BLS, or FRED integration until those commands run successfully on a machine with network access.
