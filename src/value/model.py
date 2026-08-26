from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class ValueAssumptions:
    observed_complaints: int
    hourly_wage: float
    average_handling_minutes: float = 22.0
    ai_addressable_share: float = 0.35
    time_reduction: float = 0.25
    adoption_rate: float = 0.70
    implementation_cost: float = 1_250_000.0
    annual_operating_cost: float = 350_000.0
    success_probability: float = 0.75
    discount_rate: float = 0.10


SCENARIOS = {
    "Conservative": {"ai_addressable_share": 0.22, "time_reduction": 0.15, "adoption_rate": 0.55, "success_probability": 0.60, "implementation_cost": 1.20},
    "Baseline": {"ai_addressable_share": 0.35, "time_reduction": 0.25, "adoption_rate": 0.70, "success_probability": 0.75, "implementation_cost": 1.00},
    "Aggressive": {"ai_addressable_share": 0.48, "time_reduction": 0.35, "adoption_rate": 0.82, "success_probability": 0.82, "implementation_cost": 0.92},
}


def calculate_value(assumptions: ValueAssumptions) -> dict[str, float]:
    hours_observed = assumptions.observed_complaints * assumptions.average_handling_minutes / 60
    addressable_hours = hours_observed * assumptions.ai_addressable_share
    gross_hours_released = addressable_hours * assumptions.time_reduction
    expected_hours_released = gross_hours_released * assumptions.adoption_rate * assumptions.success_probability
    capacity_value = expected_hours_released * assumptions.hourly_wage
    annual_net_value = capacity_value - assumptions.annual_operating_cost
    three_year_net_value = (annual_net_value * 3) - assumptions.implementation_cost
    total_cost = assumptions.implementation_cost + (assumptions.annual_operating_cost * 3)
    roi = np.nan if total_cost == 0 else ((capacity_value * 3) - total_cost) / total_cost
    cash_flows = [-assumptions.implementation_cost, annual_net_value, annual_net_value, annual_net_value]
    npv = sum(cf / ((1 + assumptions.discount_rate) ** period) for period, cf in enumerate(cash_flows))
    payback = np.inf if annual_net_value <= 0 else assumptions.implementation_cost / annual_net_value
    return {
        **asdict(assumptions),
        "observed_hours_workload": hours_observed,
        "ai_addressable_complaints": assumptions.observed_complaints * assumptions.ai_addressable_share,
        "gross_hours_released": gross_hours_released,
        "expected_hours_released": expected_hours_released,
        "estimated_annual_capacity_value": capacity_value,
        "annual_net_value": annual_net_value,
        "three_year_net_value": three_year_net_value,
        "roi": roi,
        "npv": npv,
        "payback_years": payback,
    }


def apply_scenario(base: ValueAssumptions, scenario: str) -> ValueAssumptions:
    factors = SCENARIOS[scenario]
    data = asdict(base)
    for key, value in factors.items():
        if key == "implementation_cost":
            data[key] = base.implementation_cost * value
        else:
            data[key] = value
    return ValueAssumptions(**data)


def sensitivity(base: ValueAssumptions, swing: float = 0.2) -> list[dict[str, float | str]]:
    outputs = []
    baseline = calculate_value(base)["three_year_net_value"]
    for field in ["ai_addressable_share", "average_handling_minutes", "time_reduction", "adoption_rate", "hourly_wage", "implementation_cost", "annual_operating_cost", "success_probability"]:
        current = getattr(base, field)
        low_data = asdict(base)
        high_data = asdict(base)
        low_data[field] = max(0, current * (1 - swing))
        high_data[field] = current * (1 + swing)
        low = calculate_value(ValueAssumptions(**low_data))["three_year_net_value"]
        high = calculate_value(ValueAssumptions(**high_data))["three_year_net_value"]
        outputs.append({"assumption": field, "low_case_delta": low - baseline, "high_case_delta": high - baseline, "range": abs(high - low)})
    return sorted(outputs, key=lambda item: item["range"], reverse=True)


def scenario_table(base: ValueAssumptions) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        rows.append(calculate_value(apply_scenario(base, scenario)) | {"scenario": scenario})
    return pd.DataFrame(rows)


def monte_carlo_value(base: ValueAssumptions, simulations: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(simulations):
        sampled = ValueAssumptions(
            observed_complaints=base.observed_complaints,
            hourly_wage=float(max(1, rng.normal(base.hourly_wage, base.hourly_wage * 0.08))),
            average_handling_minutes=float(max(1, rng.normal(base.average_handling_minutes, 5))),
            ai_addressable_share=float(rng.triangular(0.12, base.ai_addressable_share, min(0.70, base.ai_addressable_share + 0.22))),
            time_reduction=float(rng.triangular(0.08, base.time_reduction, min(0.60, base.time_reduction + 0.20))),
            adoption_rate=float(rng.triangular(0.30, base.adoption_rate, 0.95)),
            implementation_cost=float(max(0, rng.lognormal(np.log(max(base.implementation_cost, 1)), 0.25))),
            annual_operating_cost=float(max(0, rng.lognormal(np.log(max(base.annual_operating_cost, 1)), 0.20))),
            success_probability=float(rng.triangular(0.35, base.success_probability, 0.95)),
            discount_rate=base.discount_rate,
        )
        rows.append(calculate_value(sampled) | {"simulation": i + 1})
    return pd.DataFrame(rows)


def monte_carlo_summary(samples: pd.DataFrame) -> dict[str, float]:
    if samples.empty:
        return {}
    return {
        "simulations": int(len(samples)),
        "median_npv": float(samples["npv"].median()),
        "p10_npv": float(samples["npv"].quantile(0.10)),
        "p50_npv": float(samples["npv"].quantile(0.50)),
        "p90_npv": float(samples["npv"].quantile(0.90)),
        "probability_npv_positive": float((samples["npv"] > 0).mean()),
        "probability_payback_within_3_years": float((samples["payback_years"] <= 3).mean()),
    }
