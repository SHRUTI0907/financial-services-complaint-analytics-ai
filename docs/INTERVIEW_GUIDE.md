# Interview Guide

## Why this business problem?

Financial-services complaints are a real operating signal. They touch customer experience, risk, operations, compliance, product, and executive reporting. I wanted a project that felt like business analyst work: messy data, imperfect evidence, decisions under uncertainty, and a need to explain what the numbers can and cannot prove.

## Why CFPB data?

The CFPB Consumer Complaint Database is public, official, and messy enough to be realistic. It has dates, products, issues, companies, states, response fields, and some public complaint narratives. That lets the project show analytics, NLP, classification, evidence retrieval, and governance without making up companies or complaint records.

## What is the architecture?

CFPB data is ingested into a local Parquet store. The app builds deterministic analytics, data-quality artifacts, topic modeling, a routing classifier, a hybrid RAG index, value-model outputs, and governance artifacts. Streamlit is the product surface.

## Why hybrid retrieval?

Plain keyword search misses wording variation. Dense retrieval can blur exact complaint language. The project combines TF-IDF lexical retrieval, local LSA dense vectors, metadata filters, and rank fusion. That gives exact-term sensitivity plus broader semantic matching without a hosted embedding dependency.

## How does RAG work?

The index stores public complaint narratives with complaint ID, date, company, product, issue, state, and cleaned text. A user question is filtered by metadata, scored by lexical and dense retrieval, and returned with CFPB complaint citations. The answer layer receives only retrieved evidence and deterministic analytics context.

## How did you evaluate RAG?

I created 45 analyst-style questions covering products, companies, issues, date filters, state filters, comparisons, and impossible filters. The current local release measured 97.8% Recall@5, 97.8% Recall@10, and 100.0% citation validity.

## How do you stop hallucinations?

The LLM is optional. Without an API key, the app uses a deterministic fallback. The answer rules require citations, use deterministic analytics for metrics, and abstain when evidence is missing. The UI shows the retrieved complaint rows and trace so users can inspect the basis of the answer.

## How does entity resolution work?

The project keeps an accepted seed map separate from review suggestions. It uses normalized exact matching, alias suggestions, fuzzy suggestions, confidence tiers, and manual-review gates. Weak matches are not accepted automatically.

## Why use SEC assets/revenue?

Raw complaint counts can punish larger institutions just for being larger. SEC assets and revenue can provide rough public scale proxies for public registrants. They are useful context, not perfect denominators.

## Why are they imperfect denominators?

Assets and revenue do not equal customers, accounts, transaction count, geography, or product exposure. A bank may have huge assets and fewer consumer accounts, or a card issuer may have different complaint exposure than a mortgage servicer. The app labels them as scale proxies only.

## How does the value model work?

It starts with observed complaint volume. Then it applies assumptions for average handling time, AI-addressable share, time reduction, adoption, success probability, implementation cost, operating cost, and discount rate. Outputs include hours released, capacity value, net value, NPV, ROI, and payback.

## Which variables are assumptions?

Handling time, addressable share, time reduction, adoption, success probability, implementation cost, annual operating cost, and discount rate. Wage is a user assumption unless a verified BLS benchmark artifact exists.

## How does Monte Carlo work?

The model samples uncertain assumptions from documented distributions using a fixed seed. It runs 5,000 simulations in the verified release and reports median NPV, P10/P50/P90, probability NPV is positive, and probability payback happens within 3 years.

## What does macro F1 mean?

Macro F1 averages F1 across classes equally. It matters here because complaint product classes are imbalanced. Accuracy alone could look decent while ignoring smaller but important classes.

## Why this classification model?

TF-IDF + Logistic Regression is simple, fast, explainable, and appropriate for a portfolio project where the goal is analyst decision support, not deep learning spectacle. It also gives a clear baseline comparison.

## What is NMF?

Non-negative Matrix Factorization decomposes the TF-IDF document-term matrix into topics. In plain English, it groups repeated word patterns so an analyst can inspect complaint themes.

## How is AI governance implemented?

The project has an AI system inventory, risk register, controls table, and model cards. They cover the actual implemented systems: routing, topic modeling, anomaly detection, retrieval, RAG/LLM answer generation, and value/opportunity modeling.

## What NIST concepts were used?

The governance language is informed by NIST AI RMF concepts such as validity/reliability, safety, security/resilience, accountability/transparency, explainability, privacy, fairness, and human oversight. The project does not claim compliance certification.

## Biggest limitations

CFPB complaints are not a statistically complete sample. Public narratives are sparse. Entity resolution needs manual review. SEC/BLS/FRED live pulls were blocked in this sandbox and must be verified before claiming those integrations. Value estimates are scenario outputs, not realized savings.

## What would change with enterprise internal data?

I would add case resolution outcomes, customer/account exposure, transaction volume, handle time, QA labels, staffing cost, complaint severity, and internal taxonomy. That would make normalized benchmarking and value modeling much stronger.

## How would it scale from 250K to the full CFPB corpus?

Keep Parquet as the analytical layer, batch the RAG index build, use incremental refreshes, store topic/model artifacts with timestamps, and move to stronger embeddings only after measuring retrieval improvement. The architecture stays simple.
