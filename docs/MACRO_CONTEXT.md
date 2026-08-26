# Macro Context

FRED is optional. It should be used only where the selected macro series has a logical connection to the complaint trend being inspected.

Candidate official FRED series:

- `REVOLSL`: revolving consumer credit;
- `TERMCBCCALLNS`: credit-card finance rate;
- `DRCCLACBS`: credit-card loan delinquency rate;
- `UNRATE`: unemployment rate.

Current verified FRED series in this local release: 0.

Run:

```bash
export FRED_API_KEY=...
python -m src.ingestion.fred
```

Correlation does not imply causation. Macro overlays are context, not proof.
