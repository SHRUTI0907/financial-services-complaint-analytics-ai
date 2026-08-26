from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BLS_WAGE_BENCHMARKS_PATH, COMPLAINTS_PARQUET, VALUE_MONTE_CARLO_PATH, VALUE_SCENARIO_PATH
from src.value.model import ValueAssumptions, monte_carlo_summary, monte_carlo_value, scenario_table


def _default_hourly_wage() -> float:
    if BLS_WAGE_BENCHMARKS_PATH.exists():
        wages = pd.read_csv(BLS_WAGE_BENCHMARKS_PATH)
        match = wages[wages["occupation_code"].astype(str).eq("43-4051")]
        if not match.empty and pd.notna(match.iloc[0].get("mean_hourly_wage")):
            return float(match.iloc[0]["mean_hourly_wage"])
    return 24.0


def main() -> None:
    complaints = pd.read_parquet(COMPLAINTS_PARQUET)
    base = ValueAssumptions(observed_complaints=int(len(complaints)), hourly_wage=_default_hourly_wage())
    scenarios = scenario_table(base)
    samples = monte_carlo_value(base, simulations=5000, seed=42)
    VALUE_SCENARIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(VALUE_SCENARIO_PATH, index=False)
    samples.to_csv(VALUE_MONTE_CARLO_PATH, index=False)
    summary = monte_carlo_summary(samples)
    VALUE_MONTE_CARLO_PATH.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
