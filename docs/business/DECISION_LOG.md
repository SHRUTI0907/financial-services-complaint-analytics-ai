# Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-08-09 | Replace synthetic AI-use-case dataset with official CFPB complaint data. | Real-world public data makes the project more credible and interview-defensible. |
| 2026-08-09 | Use Parquet/DuckDB local analytical storage. | Supports large complaint volumes without loading everything inefficiently into memory. |
| 2026-08-09 | Use SEC EDGAR only for defensible public-company scale proxies. | Avoids simplistic raw complaint rankings while keeping limitations visible. |
| 2026-08-09 | Use BLS OEWS wages as external benchmarks, not company costs. | Enables transparent capacity modeling without inventing labor data. |
| 2026-08-09 | Start NLP with explainable TF-IDF/NMF/logistic-regression baseline. | Provides meaningful NLP/modeling while remaining reproducible and explainable. |
| 2026-08-09 | Ground governance in NIST AI RMF concepts. | Aligns AI controls to official risk-management language without claiming compliance. |
