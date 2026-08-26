from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.nlp.retrieval import retrieve_evidence


def evaluate_retrieval(complaints: pd.DataFrame, eval_set_path: Path, output_path: Path) -> pd.DataFrame:
    eval_set = pd.read_csv(eval_set_path)
    rows = []
    for _, item in eval_set.iterrows():
        evidence = retrieve_evidence(complaints, str(item["question"]), top_k=5)
        expected_terms = [term.strip().lower() for term in str(item.get("expected_evidence_terms", "")).split("|") if term.strip()]
        combined = " ".join(evidence.get("clean_narrative", pd.Series(dtype=str)).astype(str)).lower()
        rows.append(
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "retrieved_records": int(len(evidence)),
                "has_citable_complaint_ids": bool("complaint_id" in evidence and evidence["complaint_id"].notna().any()),
                "term_recall": 0 if not expected_terms else sum(term in combined for term in expected_terms) / len(expected_terms),
                "groundedness_gate": bool("complaint_id" in evidence and evidence["complaint_id"].notna().any() and len(evidence) > 0),
            }
        )
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
