# Portfolio Case Study

## Business Problem

Financial-services leaders need a practical way to see where customer friction is showing up, which issues are increasing, and where AI could support operations without creating new risk.

## Real Data

The project uses 250,000 real CFPB complaint records from 2016-03-02 to 2026-08-09. The data includes companies, products, issues, states, response fields, dates, and 5,386 public narratives.

## Analysis

The app starts with deterministic analytics: complaint volume, product mix, issue mix, company benchmarks, response behavior, and monthly trend changes. Emerging-risk scoring uses a transparent trailing-baseline z-score.

## AI / NLP

The NLP layer includes NMF topic modeling and a TF-IDF + Logistic Regression routing classifier. The classifier improved macro F1 from 0.052 baseline to 0.729 on the verified local release.

## RAG

The grounded RAG layer indexes 5,343 public complaint narratives. It retrieves cited complaint evidence with metadata filters and feeds deterministic analytics context into the answer layer. The 45-question evaluation measured 97.8% Recall@5 and 100.0% citation validity.

## Financial Value

The value model estimates capacity value under selected assumptions. It includes conservative, baseline, and aggressive scenarios, tornado sensitivity, and 5,000 Monte Carlo simulations. The app labels this as modelled capacity value, not realized savings.

## Governance

The project governs six AI systems: routing classifier, topic model, emerging-risk detector, hybrid retrieval, grounded LLM answer layer, and AI value/opportunity model. It documents 14 risks and 12 controls using NIST AI RMF concepts without claiming certification.

## Executive Decision

The final product feels like an enterprise analytics and AI decision-support workspace: it gives leaders the data, the evidence, the model caveats, the value assumptions, and the governance controls in one place.
