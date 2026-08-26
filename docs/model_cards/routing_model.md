# Model Card: Complaint-Routing Classifier

## Intended Use
Decision support for product-level complaint routing analysis.

## Out Of Scope
Autonomous customer decisions, regulatory findings, or final complaint disposition.

## Data
Public CFPB complaint narratives from the local 250K-record extract.

## Method
TF-IDF features with Logistic Regression.

## Evaluation
Macro F1 is measured against a most-frequent baseline. Current macro F1 is documented in `data/artifacts/model_report.json`.

## Limitations
Narratives are sparse, public text is redacted, product categories drift, and class balance is uneven.

## Monitoring
Macro F1, per-class precision/recall, confusion matrix, class drift, low-confidence routing volume.
