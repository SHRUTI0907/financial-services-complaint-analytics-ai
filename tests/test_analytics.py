from __future__ import annotations

import unittest

import pandas as pd

from src.analytics.anomaly import emerging_issue_score
from src.analytics.benchmarking import company_benchmark
from src.analytics.complaints import kpi_summary, volume_by_month


def fixture() -> pd.DataFrame:
    rows = []
    complaint_id = 1
    for month in pd.date_range("2024-01-01", periods=15, freq="MS"):
        count = 20 if month < pd.Timestamp("2025-03-01") else 80
        for _ in range(count):
            rows.append(
                {
                    "complaint_id": complaint_id,
                    "date_received": month,
                    "product": "Credit card",
                    "issue": "Payment problem",
                    "company": "Mapped Bank",
                    "state": "NY",
                    "submitted_via": "Web",
                    "company_response_to_consumer": "Closed with explanation",
                    "timely_response": "Yes",
                    "consumer_complaint_narrative": "The payment was not applied to my account and the company did not explain the delay.",
                }
            )
            complaint_id += 1
    return pd.DataFrame(rows)


class AnalyticsTests(unittest.TestCase):
    def test_kpi_summary(self) -> None:
        summary = kpi_summary(fixture())
        self.assertGreater(summary["complaints"], 0)
        self.assertEqual(summary["products"], 1)

    def test_volume_by_month(self) -> None:
        monthly = volume_by_month(fixture())
        self.assertEqual(monthly["month"].nunique(), 15)

    def test_emerging_issue_score_has_latest_month(self) -> None:
        emerg = emerging_issue_score(fixture(), "issue")
        self.assertFalse(emerg.empty)
        self.assertIn("emerging_issue_score", emerg.columns)

    def test_company_benchmark_normalizes_when_scale_available(self) -> None:
        scale = pd.DataFrame(
            {
                "cfpb_company_name": ["Mapped Bank"],
                "canonical_company_name": ["Mapped Bank Holding"],
                "assets": [100_000_000_000],
                "revenue": [10_000_000_000],
            }
        )
        bench = company_benchmark(fixture(), scale)
        self.assertIn("complaints_per_1b_assets", bench.columns)
        self.assertGreater(bench.loc[0, "complaints_per_1b_assets"], 0)


if __name__ == "__main__":
    unittest.main()
