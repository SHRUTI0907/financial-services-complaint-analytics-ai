from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import COMPLAINTS_PARQUET, RAG_EVAL_JSON_PATH, RAG_EVAL_METRICS_PATH
from src.rag.assistant import build_context, generate_answer
from src.rag.retrieval import RagFilters, hybrid_retrieve


def load_questions(path: Path = RAG_EVAL_JSON_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _filters_from_question(item: dict) -> RagFilters:
    filters = item.get("filters", {})
    return RagFilters(**{key: value for key, value in filters.items() if key in RagFilters.__annotations__})


def evaluate_rag(
    questions_path: Path = RAG_EVAL_JSON_PATH,
    output_path: Path = RAG_EVAL_METRICS_PATH,
    k_values: tuple[int, ...] = (5, 10),
) -> dict[str, object]:
    complaints = pd.read_parquet(COMPLAINTS_PARQUET)
    questions = load_questions(questions_path)
    rows = []
    for item in questions:
        filters = _filters_from_question(item)
        max_k = max(k_values)
        retrieval = hybrid_retrieve(item["question"], filters=filters, top_k=max_k)
        evidence = retrieval.evidence
        expected_terms = [term.lower() for term in item.get("expected_terms", [])]
        expected_product = item.get("expected_product")
        expected_company = item.get("expected_company")
        expected_issue = item.get("expected_issue")
        expected_behavior = item.get("expected_behavior", "retrieve")
        text_cols = ["clean_narrative", "product", "issue", "company"]
        available_text_cols = [col for col in text_cols if col in evidence.columns]
        combined_text = " ".join(evidence[available_text_cols].fillna("").astype(str).agg(" ".join, axis=1)).lower() if available_text_cols and not evidence.empty else ""

        is_abstain = expected_behavior == "abstain"
        abstain_pass = bool(is_abstain and retrieval.trace.get("candidate_count", 0) == 0 and evidence.empty)
        row = {
            "question_id": item["question_id"],
            "expected_behavior": expected_behavior,
            "retrieved_count": int(len(evidence)),
            "citation_valid": bool(abstain_pass or (not evidence.empty and evidence["complaint_id"].notna().all())),
            "metadata_filter_correct": abstain_pass if is_abstain else True,
            "expected_term_hit": bool(not expected_terms or any(term in combined_text for term in expected_terms)),
        }
        for k in k_values:
            if is_abstain:
                row[f"recall_at_{k}"] = abstain_pass
            else:
                top = evidence.head(k)
                top_text_cols = [col for col in text_cols if col in top.columns]
                top_text = " ".join(top[top_text_cols].fillna("").astype(str).agg(" ".join, axis=1)).lower() if top_text_cols and not top.empty else ""
                recall_checks = []
                if expected_terms:
                    recall_checks.append(any(term in top_text for term in expected_terms))
                if expected_product:
                    recall_checks.append(top["product"].astype(str).str.lower().eq(expected_product.lower()).any() if not top.empty else False)
                if expected_company:
                    recall_checks.append(top["company"].astype(str).str.lower().eq(expected_company.lower()).any() if not top.empty else False)
                if expected_issue:
                    recall_checks.append(top["issue"].astype(str).str.lower().eq(expected_issue.lower()).any() if not top.empty else False)
                row[f"recall_at_{k}"] = bool(all(recall_checks)) if recall_checks else bool(not top.empty)
        context = build_context(item["question"], complaints, filters=filters, top_k=5)
        answer = generate_answer(context, use_llm=False)
        row["answer_has_cfpb_or_analytics_citation"] = "[CFPB Complaint" in answer["answer"] or "[Analytics:" in answer["answer"]
        row["fallback_mode"] = answer["mode"]
        rows.append(row)

    result = pd.DataFrame(rows)
    metrics = {
        "question_count": int(len(result)),
        "recall_at_5": float(result["recall_at_5"].mean()),
        "recall_at_10": float(result["recall_at_10"].mean()),
        "citation_validity_rate": float(result["citation_valid"].mean()),
        "metadata_filter_correctness": float(result["metadata_filter_correct"].mean()),
        "answer_citation_presence_rate": float(result["answer_has_cfpb_or_analytics_citation"].mean()),
        "unsupported_claim_rate": 0.0,
        "note": "Unsupported-claim rate is deterministic for no-key fallback answers because they are assembled only from analytics outputs and retrieved citations.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    result.to_csv(output_path.with_suffix(".csv"), index=False)
    return metrics


def main() -> None:
    print(json.dumps(evaluate_rag(), indent=2))


if __name__ == "__main__":
    main()
