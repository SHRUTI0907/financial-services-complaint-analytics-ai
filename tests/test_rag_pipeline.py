from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.rag.analytics_tools import get_complaint_volume, get_growth_rate, get_top_issues
from src.rag.assistant import build_context, generate_answer
from src.rag.index import RagIndexConfig, build_rag_index
from src.rag.retrieval import RagFilters, hybrid_retrieve


def rag_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "complaint_id": 1,
                "date_received": "2026-01-01",
                "company": "Alpha Bank",
                "product": "Credit card",
                "sub_product": "General-purpose credit card",
                "issue": "Problem with a purchase",
                "sub_issue": "Card payment issue",
                "state": "NY",
                "consumer_complaint_narrative": "My credit card payment was not credited and I received a late fee even though I paid on time.",
                "submitted_via": "Web",
                "company_response_to_consumer": "Closed with explanation",
                "timely_response": "Yes",
            },
            {
                "complaint_id": 2,
                "date_received": "2026-02-01",
                "company": "Beta Servicing",
                "product": "Mortgage",
                "sub_product": "Conventional home mortgage",
                "issue": "Trouble during payment process",
                "sub_issue": "Escrow issue",
                "state": "CA",
                "consumer_complaint_narrative": "The mortgage servicer calculated escrow incorrectly and did not respond to repeated written requests.",
                "submitted_via": "Web",
                "company_response_to_consumer": "Closed with explanation",
                "timely_response": "Yes",
            },
            {
                "complaint_id": 3,
                "date_received": "2026-03-01",
                "company": "Gamma Collections",
                "product": "Debt collection",
                "sub_product": "Credit card debt",
                "issue": "Attempts to collect debt not owed",
                "sub_issue": "Debt is not yours",
                "state": "TX",
                "consumer_complaint_narrative": "The collector kept asking me to pay a debt I do not owe and would not provide validation.",
                "submitted_via": "Web",
                "company_response_to_consumer": "Closed with explanation",
                "timely_response": "No",
            },
        ]
    )


class RagPipelineTests(unittest.TestCase):
    def test_hybrid_retrieval_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "complaints.parquet"
            index = root / "index.joblib"
            metadata = root / "metadata.parquet"
            rag_fixture().to_parquet(parquet, index=False)
            build_rag_index(parquet, index, metadata, RagIndexConfig(dense_components=2))
            result = hybrid_retrieve("late fee card payment", RagFilters(product="Credit card"), top_k=2, index_path=index)
            self.assertEqual(len(result.evidence), 1)
            self.assertEqual(result.evidence.loc[0, "complaint_id"], 1)
            self.assertIn("[CFPB Complaint 1]", result.evidence.loc[0, "citation"])

    def test_analytics_tools_are_deterministic(self) -> None:
        df = rag_fixture()
        filters = RagFilters(product="Debt collection")
        self.assertEqual(get_complaint_volume(df, filters)["complaints"], 1)
        self.assertEqual(get_top_issues(df, filters)["rows"][0]["issue"], "Attempts to collect debt not owed")
        self.assertIn("metric_id", get_growth_rate(df, filters))

    def test_no_key_answer_fallback_has_citations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "complaints.parquet"
            index = root / "index.joblib"
            metadata = root / "metadata.parquet"
            df = rag_fixture()
            df.to_parquet(parquet, index=False)
            build_rag_index(parquet, index, metadata, RagIndexConfig(dense_components=2))
            from unittest.mock import patch

            with patch("src.rag.assistant.RAG_INDEX_PATH", index):
                context = build_context("What happened with card payments?", df, RagFilters(product="Credit card"))
                answer = generate_answer(context, use_llm=False)
            self.assertEqual(answer["mode"], "deterministic_no_key_fallback")
            self.assertIn("[CFPB Complaint 1]", answer["answer"])
            self.assertIn("[Analytics:", answer["answer"])
            self.assertIn("evidence", answer)
            self.assertEqual(len(answer["evidence"]), 1)


if __name__ == "__main__":
    unittest.main()
