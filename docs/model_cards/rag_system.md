# Model Card: Grounded RAG System

## Intended Use
Retrieve complaint evidence and draft cited analyst answers.

## Out Of Scope
Autonomous decisions, misconduct claims, or uncited executive claims.

## Data
5,343 indexed CFPB public narratives with metadata.

## Method
TF-IDF lexical retrieval, local LSA dense retrieval, metadata filtering, rank fusion, deterministic analytics context, optional LLM generation.

## Evaluation
45 RAG questions; Recall@5 97.8%; citation validity 100.0%.

## Limitations
Only public narratives are searchable. Ambiguous prompts can retrieve semantically adjacent records. The LLM is optional and must be reviewed.

## Monitoring
Recall@K, citation validity, unsupported-claim rate, abstention rate, retrieval failure review.
