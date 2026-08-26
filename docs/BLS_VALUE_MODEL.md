# BLS Value Model

The value model separates observed workload, external benchmarks, user assumptions, and model output.

## Observed Input

Real CFPB complaint volume from the processed Parquet store.

## External Benchmark

BLS OEWS wage benchmarks can be loaded with:

```bash
python -m src.ingestion.bls_oews
```

Current verified BLS benchmark rows in this local release: 0.

If no BLS artifact is present, the UI clearly labels wage as a user assumption.

## User Assumptions

Average handling time, AI-addressable share, time reduction, adoption rate, success probability, implementation cost, annual operating cost, discount rate.

## Model Output

Hours potentially released, estimated annual capacity value, annual net value, three-year value, NPV, ROI, payback, scenario results, Monte Carlo distribution.

Use this wording: "Estimated annual capacity value under selected assumptions." Do not write "AI saves X."
