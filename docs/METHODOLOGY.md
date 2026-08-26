# Methodology

## Data Engineering
The project stores large CFPB complaint data locally in Parquet and queries analytical views through DuckDB where available. Expensive files such as topic registries, model reports, and opportunity catalogs are built by preprocessing scripts rather than recomputed on app startup.

## Data Quality
Validation checks include:
- required CFPB fields;
- unique complaint IDs;
- parseable complaint dates;
- missing-value profiling;
- narrative availability;
- response and submission-channel distributions.

The CFPB product and issue taxonomy can change over time. Longitudinal analysis should inspect historical taxonomy changes before applying category mappings.

## Time-Series Intelligence
Complaint spikes are detected by comparing latest monthly volume to a trailing 12-month baseline. A z-score of 2.0 or higher is flagged as a spike when enough history exists. This indicates statistical unusualness, not causation.

## NLP
The baseline NLP pipeline uses:
- redaction-aware text cleaning;
- TF-IDF features;
- NMF topic discovery;
- TF-IDF retrieval for citable complaint evidence;
- logistic-regression classification for routing/product prediction.

This approach is intentionally explainable and can be upgraded to sentence-transformer embeddings, BERTopic, UMAP, or HDBSCAN when compute and dependency constraints justify it.

## AI Opportunity Discovery
AI opportunities are derived from observed complaint product/issue patterns and narrative evidence. Each opportunity separates observed evidence from proposed intervention. Examples include complaint routing, agent-assist retrieval, summarization, root-cause detection, and emerging-risk monitoring.

## Value Realization
The model estimates capacity value, not realized savings:

`observed hours = observed complaints * average handling minutes / 60`

`expected hours released = observed hours * AI-addressable share * time reduction * adoption * success probability`

`estimated annual capacity value = expected hours released * hourly wage benchmark`

`annual net value = estimated annual capacity value - annual operating cost`

`3-year net value = annual net value * 3 - implementation cost`

`ROI = (3-year capacity value - total 3-year cost) / total 3-year cost`

`NPV = discounted implementation outflow and annual net-value cash flows`

## Governance
The governance model is influenced by the official NIST AI RMF and GenAI Profile concepts: validity/reliability, safety, security/resilience, accountability/transparency, explainability/interpretability, privacy, fairness, and human oversight. The controls are project-specific recommendations and do not certify compliance.
