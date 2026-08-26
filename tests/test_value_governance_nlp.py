from __future__ import annotations

import unittest

import pandas as pd

from src.governance.nist import governance_catalog, risk_tier
from src.nlp.text import clean_text, narrative_frame
from src.value.model import ValueAssumptions, apply_scenario, calculate_value, sensitivity
from src.value.opportunities import derive_ai_opportunities


class ValueGovernanceNlpTests(unittest.TestCase):
    def test_value_model_zero_adoption(self) -> None:
        assumptions = ValueAssumptions(observed_complaints=1000, hourly_wage=25, adoption_rate=0)
        result = calculate_value(assumptions)
        self.assertEqual(result["expected_hours_released"], 0)
        self.assertLess(result["annual_net_value"], 0)

    def test_scenario_changes_assumptions(self) -> None:
        base = ValueAssumptions(observed_complaints=1000, hourly_wage=25)
        conservative = apply_scenario(base, "Conservative")
        self.assertLess(conservative.adoption_rate, base.adoption_rate)
        self.assertGreater(conservative.implementation_cost, base.implementation_cost)

    def test_sensitivity_outputs_ranked_ranges(self) -> None:
        result = sensitivity(ValueAssumptions(observed_complaints=2000, hourly_wage=25))
        self.assertGreater(len(result), 3)
        self.assertGreaterEqual(result[0]["range"], result[-1]["range"])

    def test_clean_text_removes_common_redaction_artifacts(self) -> None:
        self.assertNotIn("XXXX", clean_text("I paid XXXX on xx/xx/2025 and got no response."))

    def test_narrative_frame_filters_short_text(self) -> None:
        df = pd.DataFrame({"consumer_complaint_narrative": ["short", "This is a long enough complaint narrative about payment routing problems and unexplained delays."]})
        self.assertEqual(len(narrative_frame(df, min_chars=20)), 1)

    def test_opportunities_derive_from_observed_rows(self) -> None:
        df = pd.DataFrame(
            {
                "product": ["Credit card", "Credit card", "Checking"] * 4,
                "issue": ["Fee problem", "Payment problem", "Account blocked"] * 4,
            }
        )
        opps = derive_ai_opportunities(df)
        self.assertFalse(opps.empty)
        self.assertIn("observed_problem", opps.columns)

    def test_governance_catalog_adds_controls(self) -> None:
        opps = pd.DataFrame(
            [
                {
                    "opportunity_name": "Complaint summarization",
                    "potential_ai_intervention": "Summarize complaints with citations.",
                    "risk_considerations": "Misrepresentation and hallucination.",
                }
            ]
        )
        catalog = governance_catalog(opps)
        self.assertEqual(risk_tier("Complaint summarization"), "Moderate")
        self.assertIn("source-grounded", catalog.loc[0, "recommended_controls"])


if __name__ == "__main__":
    unittest.main()
