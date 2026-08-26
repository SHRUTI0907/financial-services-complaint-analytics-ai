# Business Requirements

## Problem Statement
Financial-services leaders need a defensible way to identify operational friction, consumer-risk signals, and AI intervention opportunities from real complaint data. Existing analysis is often spreadsheet-led, reactive, and disconnected from value modeling and AI governance.

## Objectives
- Ingest official public complaint data at scale.
- Surface product, issue, company, geography, and response patterns.
- Detect emerging complaint risks before they are obvious in static reports.
- Discover NLP themes from public narratives.
- Benchmark public companies using transparent scale proxies.
- Estimate AI-assisted operational capacity value under editable assumptions.
- Link proposed AI use cases to governance controls.

## Stakeholders
CFO, COO, Chief Data/AI Officer, CIO, Risk, Legal, Compliance, Operations, Product, Data Science, Customer Experience, and frontline complaint teams.

## Functional Requirements
- Refresh CFPB complaint data from official sources.
- Profile data quality and missingness.
- Filter by company, product, issue, state, response, channel, and date.
- Detect spikes using documented thresholds.
- Build topic and retrieval artifacts from public narratives.
- Train and evaluate at least one practical predictive model.
- Map selected CFPB companies to SEC registrants with confidence and review flags.
- Toggle raw and normalized company benchmarking views.
- Model value using observed volume, BLS wage benchmarks, and user assumptions.
- Produce NIST-aligned governance recommendations.

## Non-Functional Requirements
- Reproducible local pipeline.
- No large raw data committed to Git.
- Cached preprocessing for app responsiveness.
- Transparent limitations and assumptions.
- Clear separation of observed evidence and proposed intervention.

## Business Rules
- Never treat complaint count as a complete measure of company performance.
- Never force entity matches.
- Never claim realized savings from scenario estimates.
- Never claim causation from correlations or spikes.
- Never claim regulatory compliance from suggested controls.

## Acceptance Criteria
- App opens without data and clearly instructs official ingestion.
- Pipeline writes a Parquet complaint store and metadata.
- Tests pass locally.
- Value model labels observed inputs, external benchmarks, assumptions, and outputs.
- Governance pages show controls tied to implemented AI use cases.
