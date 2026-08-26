# Entity Resolution

CFPB company names are operational names, brands, servicers, banks, subsidiaries, and parent-company labels. The project does not automatically merge weak matches.

## Current Artifacts

- Distinct CFPB companies: 1,383
- Accepted seed mappings: 15
- High-confidence mappings: 10
- Medium-review mappings: 5
- Review artifact: `data/artifacts/entity_resolution_review.csv`

## Matching Layers

1. Normalized exact matching.
2. Alias suggestions for known brands/subsidiaries.
3. Fuzzy suggestions with manual-review threshold.
4. Weak matches are rejected or sent to review.

## Rule

Do not show normalized SEC metrics unless the mapping is high confidence, no manual review is required, the financial denominator exists, and the denominator is nonzero.
