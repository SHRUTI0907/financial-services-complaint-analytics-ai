# Final Verified Metrics

These are the verified metrics for the current release.

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
| Classification classes | 10 | `data/artifacts/model_report.json` |
| Classification macro F1 | 0.729 | `data/artifacts/model_report.json` |
| Baseline macro F1 | 0.052 | `data/artifacts/model_report.json` |
| RAG evaluation questions | 45 | `evaluation/rag_questions.json` |
| RAG Recall@5 | 97.8% | `data/artifacts/rag_eval_metrics.json` |
| RAG Recall@10 | 97.8% | `data/artifacts/rag_eval_metrics.json` |
| Citation validity | 100.0% | `data/artifacts/rag_eval_metrics.json` |
| Live dashboard sections | 5 | `app.py` navigation |
| Tests passed | 25 / 25 | `python3 -m unittest discover -s tests` |

## Resume-Safe Summary

- Analyzed 250K real CFPB complaints across 1,383 companies.
- Evaluated 12 products and 87 issue categories using trend and anomaly analysis.
- Built NLP and grounded RAG features over 5,343 indexed complaint narratives.
- Developed a 10-class complaint-routing model with 0.729 macro F1.
