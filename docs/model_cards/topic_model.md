# Model Card: NMF Topic Model

## Intended Use
Explore repeated language in public complaint narratives.

## Out Of Scope
Proof of root cause, misconduct, or market-wide prevalence.

## Data
Cleaned public CFPB complaint narratives.

## Method
TF-IDF vectorization and NMF topic modeling.

## Evaluation
Topic rows and volumes are stored in `data/artifacts/topic_registry.csv`; topics require analyst review.

## Limitations
Topic labels are interpretive, templates can dominate, and topics can shift after refresh.

## Monitoring
Topic volume drift, top-term stability, example narrative review, SME feedback.
