from __future__ import annotations

import unittest

import pandas as pd

from src.ingestion.cfpb import normalize_columns
from src.quality.profile import build_quality_profile, validate_complaints_schema


class QualityAndIngestionTests(unittest.TestCase):
    def test_normalize_columns_preserves_official_fields(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "Date received": "2025-01-01",
                    "Product": "Credit card",
                    "Issue": "Problem with a purchase",
                    "Company": "Example Bank",
                    "Complaint ID": "123",
                }
            ]
        )
        normalized = normalize_columns(raw)
        self.assertIn("date_received", normalized.columns)
        self.assertIn("complaint_id", normalized.columns)
        self.assertEqual(int(normalized.loc[0, "complaint_id"]), 123)
        self.assertTrue(pd.isna(normalized.loc[0, "sub_product"]))

    def test_validation_catches_duplicate_ids(self) -> None:
        df = pd.DataFrame(
            {
                "complaint_id": [1, 1],
                "date_received": ["2025-01-01", "2025-01-02"],
                "product": ["A", "A"],
                "issue": ["I", "I"],
                "company": ["C", "C"],
                "state": ["NY", "NY"],
                "submitted_via": ["Web", "Web"],
                "company_response_to_consumer": ["Closed", "Closed"],
                "timely_response": ["Yes", "No"],
                "consumer_complaint_narrative": ["Long enough narrative", None],
            }
        )
        errors = validate_complaints_schema(df)
        self.assertTrue(any("Duplicate" in error for error in errors))

    def test_quality_profile_reports_missingness(self) -> None:
        df = pd.DataFrame({"a": [1, None, 3]})
        profile = build_quality_profile(df)
        self.assertEqual(profile.loc[0, "missing_count"], 1)
        self.assertAlmostEqual(profile.loc[0, "missing_rate"], 0.3333)


if __name__ == "__main__":
    unittest.main()
