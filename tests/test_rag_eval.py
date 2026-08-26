from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.nlp.rag_eval import evaluate_retrieval


class RagEvalTests(unittest.TestCase):
    def test_evaluate_retrieval_writes_report(self) -> None:
        complaints = pd.DataFrame(
            {
                "complaint_id": [1, 2],
                "date_received": ["2025-01-01", "2025-01-02"],
                "company": ["A", "B"],
                "product": ["Credit card", "Bank account"],
                "issue": ["Payment issue", "Account closed"],
                "consumer_complaint_narrative": [
                    "My credit card payment was not credited and I received a late fee after paying on time.",
                    "My checking account was closed after a fraud alert and support did not explain the decision.",
                ],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            eval_path = Path(tmp) / "eval.csv"
            out_path = Path(tmp) / "report.csv"
            pd.DataFrame(
                [{"question_id": "Q1", "question": "payment late fee", "expected_evidence_terms": "payment|late fee"}]
            ).to_csv(eval_path, index=False)
            result = evaluate_retrieval(complaints, eval_path, out_path)
            self.assertTrue(out_path.exists())
            self.assertTrue(result.loc[0, "groundedness_gate"])


if __name__ == "__main__":
    unittest.main()
