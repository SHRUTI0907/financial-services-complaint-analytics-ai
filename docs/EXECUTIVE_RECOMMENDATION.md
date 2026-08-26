# Executive Recommendation

## Executive Summary
The platform processed 250,000 real CFPB complaint records covering 2016-03-02 to 2026-08-09. The leading product category in this extract is `Credit reporting or other personal consumer reports` with 225,645 complaints. The leading issue is `Incorrect information on your report` with 133,076 complaints.

These findings come from official CFPB data and local analysis files. Because this run is capped, read it as a strong project extract, not a final report on the whole market.

## Observed Consumer Problems
- Credit reporting or other personal consumer reports: 225,645 complaints (90.3%).
- Debt collection: 11,432 complaints (4.6%).
- Credit card: 3,580 complaints (1.4%).
- Checking or savings account: 3,565 complaints (1.4%).
- Mortgage: 1,432 complaints (0.6%).
- Money transfer, virtual currency, or money service: 1,423 complaints (0.6%).
- Vehicle loan or lease: 902 complaints (0.4%).
- Student loan: 870 complaints (0.3%).

Top issues:
- Incorrect information on your report: 133,076 complaints (53.2%).
- Improper use of your report: 48,987 complaints (19.6%).
- Problem with a company's investigation into an existing problem: 42,415 complaints (17.0%).
- Attempts to collect debt not owed: 5,528 complaints (2.2%).
- Took or threatened to take negative or legal action: 2,477 complaints (1.0%).
- Managing an account: 2,020 complaints (0.8%).

## Emerging Risks
`Incorrect information on your report` is the highest latest-month emerging-issue signal in this extract, with 34,118 complaints and a z-score of 1.37.

The emerging-risk monitor uses a trailing-baseline z-score. A spike is a statistical investigation signal, not a causal conclusion.

## Company / Product Insight
`TRANSUNION INTERMEDIATE HOLDINGS, INC.` has the highest raw complaint volume in this selected extract with 87,099 complaints. The app deliberately avoids saying this means the company is worse; raw complaint volume must be interpreted with scale, product mix, customer base, and reporting limitations.

## AI Opportunity Areas
- Agent-assist retrieval: High complaint volume for Credit reporting or other personal consumer reports / Incorrect information on your report (132,587 supporting complaints).
- Root-cause detection: High complaint volume for Credit reporting or other personal consumer reports / Improper use of your report (48,948 supporting complaints).
- Root-cause detection: High complaint volume for Credit reporting or other personal consumer reports / Problem with a company's investigation into an existing problem (42,198 supporting complaints).
- Operational intelligence workbench: High complaint volume for Debt collection / Attempts to collect debt not owed (5,528 supporting complaints).
- Operational intelligence workbench: High complaint volume for Debt collection / Took or threatened to take negative or legal action (2,477 supporting complaints).
- Agent-assist retrieval: High complaint volume for Checking or savings account / Managing an account (2,020 supporting complaints).

## Estimated Value Under Explicit Assumptions
Using the baseline scenario across the selected 250,000 observed complaints:
- Estimated annual capacity value: $101.1K.
- Estimated 3-year net value: $-2.0M.
- Estimated hours released: 4,211.

These are capacity-value estimates, not realized savings. The app exposes handling time, addressable share, time reduction, adoption, wage benchmark, implementation cost, operating cost, success probability, and discount rate.

## Governance Requirements
The highest-priority controls are source-grounded generation, human review for customer-impacting decisions, model evaluation by class, drift monitoring, audit logging, privacy controls, and low-confidence escalation. NIST AI RMF concepts inform the structure, but this project does not claim compliance certification.

## Recommended Priorities
1. Use the app to monitor high-volume credit-reporting and investigation-related complaint categories.
2. Pilot human-in-the-loop complaint routing or agent-assist retrieval before any autonomous customer-impacting workflow.
3. Use topic and retrieval evidence to validate repeated failure modes with operations and compliance SMEs.
4. Run SEC and BLS enrichment for stronger normalized benchmarking and value-model calibration.

## 90-Day Actions
- Run full CFPB ingestion without the record cap.
- Review the topic registry for duplicate or template-like complaint language.
- Expand entity-resolution mappings only where a public registrant match is defensible.
- Define a governed pilot for one low-risk AI support workflow.

## 6-12 Month Roadmap
- Add internal complaint resolution outcomes and QA labels in a real enterprise deployment.
- Replace public-only denominators with customer/account/transaction exposure where available.
- Move from baseline NLP to embeddings or BERTopic after measuring topic quality gains.
- Establish production monitoring, review workflows, and governance sign-off gates.
