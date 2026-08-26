# SEC Enrichment

The SEC layer uses the official SEC EDGAR Company Facts API as the primary source. It is designed to add public-company scale context to CFPB company complaint analytics without pretending that assets or revenue equal customer count.

## Implemented

- Company Facts fetcher with required `SEC_USER_AGENT` validation.
- CIK-padded SEC endpoint handling.
- Concept-resolution logic for assets, revenue, net income, and operating income.
- Fiscal-year lineage with company, CIK, fiscal year, period, form, filed date, XBRL concept, unit, value, accession, and retrieval date.
- Scale table with `normalization_allowed` gate.

## Not Live-Verified Here

Live SEC requests were blocked by this sandbox. Run:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
python scripts/build_entity_resolution.py
python -m src.ingestion.sec_edgar --max-companies 10
```

Only use normalized metrics after `data/artifacts/sec_financial_lineage.csv` and `data/artifacts/sec_company_scale_metrics.csv` contain real SEC observations.
