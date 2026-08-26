from __future__ import annotations

import unittest

import pandas as pd

from src.nlp.retrieval import retrieve_evidence
from src.nlp.topics import discover_topics


class RetrievalAndTopicTests(unittest.TestCase):
    def test_retrieve_evidence_returns_traceable_rows(self) -> None:
        df = pd.DataFrame(
            {
                "complaint_id": [1, 2, 3],
                "date_received": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "company": ["A", "B", "C"],
                "product": ["Credit card", "Mortgage", "Bank account"],
                "issue": ["Payment issue", "Escrow issue", "Account closed"],
                "consumer_complaint_narrative": [
                    "My credit card payment was not credited and I was charged a late fee despite paying on time.",
                    "The mortgage escrow calculation was wrong and the servicer did not answer after repeated calls and written requests.",
                    "My checking account was closed without an explanation after a fraud alert and I could not reach support for weeks.",
                ],
            }
        )
        result = retrieve_evidence(df, "late fee credit card payment", top_k=2)
        self.assertEqual(len(result), 2)
        self.assertIn("complaint_id", result.columns)

    def test_topic_discovery_skips_when_too_little_text(self) -> None:
        df = pd.DataFrame({"consumer_complaint_narrative": ["short"] * 5})
        topics, docs = discover_topics(df, n_topics=3)
        self.assertTrue(topics.empty)
        self.assertIn("topic_id", docs.columns)


if __name__ == "__main__":
    unittest.main()
