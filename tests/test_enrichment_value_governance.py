from __future__ import annotations

import unittest

import pandas as pd

from src.enrichment.entity_resolution import confidence_from_score, normalize_company_name
from src.governance.inventory import ai_system_inventory, governance_controls, risk_register
from src.ingestion.bls_oews import parse_oews_workbook
from src.ingestion.sec_edgar import latest_scale_table, resolve_financial_facts
from src.value.model import ValueAssumptions, monte_carlo_summary, monte_carlo_value


class EnrichmentValueGovernanceTests(unittest.TestCase):
    def test_company_name_normalization_and_confidence(self) -> None:
        self.assertEqual(normalize_company_name("Bank of America, National Association"), "america")
        high = confidence_from_score(0.98, "manual_seed_sec_cik_ticker")
        medium = confidence_from_score(0.90, "fuzzy_suggestion")
        low = confidence_from_score(0.50, "fuzzy_suggestion")
        self.assertEqual(high.confidence_tier, "HIGH CONFIDENCE")
        self.assertFalse(high.manual_review_required)
        self.assertEqual(medium.confidence_tier, "MEDIUM - REVIEW")
        self.assertTrue(low.manual_review_required)

    def test_sec_companyfacts_resolution_and_scale_gate(self) -> None:
        facts = {
            "cik": "19617",
            "entityName": "JPMorgan Chase & Co.",
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "units": {
                            "USD": [
                                {"fy": 2023, "fp": "FY", "form": "10-K", "end": "2023-12-31", "filed": "2024-02-16", "val": 3_875_393_000_000, "accn": "a"},
                                {"fy": 2024, "fp": "FY", "form": "10-K", "end": "2024-12-31", "filed": "2025-02-14", "val": 4_002_814_000_000, "accn": "b"},
                            ]
                        }
                    },
                    "Revenues": {"units": {"USD": [{"fy": 2024, "fp": "FY", "form": "10-K", "end": "2024-12-31", "filed": "2025-02-14", "val": 177_556_000_000, "accn": "b"}]}},
                }
            },
        }
        lineage = resolve_financial_facts(facts, retrieved_at_utc="2026-01-01T00:00:00+00:00")
        self.assertIn("xbrl_concept", lineage.columns)
        self.assertEqual(int(lineage[lineage["metric_name"].eq("assets")].iloc[-1]["fiscal_year"]), 2024)
        entity_map = pd.DataFrame(
            [
                {
                    "cfpb_company_name": "JPMORGAN CHASE & CO.",
                    "canonical_company_name": "JPMorgan Chase & Co.",
                    "sec_cik": "0000019617",
                    "ticker": "JPM",
                    "confidence_tier": "HIGH CONFIDENCE",
                    "manual_review_required": False,
                },
                {
                    "cfpb_company_name": "BANK OF AMERICA, NATIONAL ASSOCIATION",
                    "canonical_company_name": "Bank of America Corporation",
                    "sec_cik": "0000070858",
                    "ticker": "BAC",
                    "confidence_tier": "MEDIUM - REVIEW",
                    "manual_review_required": True,
                },
            ]
        )
        scale = latest_scale_table(lineage, entity_map)
        self.assertTrue(bool(scale.loc[scale["ticker"].eq("JPM"), "normalization_allowed"].iloc[0]))
        self.assertFalse(bool(scale.loc[scale["ticker"].eq("BAC"), "normalization_allowed"].iloc[0]))

    def test_bls_oews_parser(self) -> None:
        raw = pd.DataFrame(
            {
                "OCC_CODE": ["43-4051", "99-9999"],
                "OCC_TITLE": ["Customer Service Representatives", "Other"],
                "H_MEAN": ["23.50", "10.00"],
                "A_MEAN": ["48,880", "20,800"],
                "H_MEDIAN": ["21.20", "9.00"],
                "A_MEDIAN": ["44,090", "18,720"],
            }
        )
        parsed = parse_oews_workbook(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed.iloc[0]["occupation_code"], "43-4051")
        self.assertAlmostEqual(float(parsed.iloc[0]["mean_hourly_wage"]), 23.50)

    def test_monte_carlo_is_reproducible(self) -> None:
        base = ValueAssumptions(observed_complaints=1000, hourly_wage=25.0)
        a = monte_carlo_value(base, simulations=100, seed=7)
        b = monte_carlo_value(base, simulations=100, seed=7)
        self.assertTrue(a["npv"].equals(b["npv"]))
        summary = monte_carlo_summary(a)
        self.assertEqual(summary["simulations"], 100)
        self.assertIn("probability_npv_positive", summary)

    def test_governance_artifacts_are_component_specific(self) -> None:
        inventory = ai_system_inventory()
        risks = risk_register()
        controls = governance_controls()
        self.assertGreaterEqual(len(inventory), 6)
        self.assertIn("Hybrid retrieval", set(inventory["system_name"]))
        self.assertGreaterEqual(len(risks), 10)
        self.assertTrue((controls["system"] == "Grounded LLM answer layer").any())


if __name__ == "__main__":
    unittest.main()
