from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analytics.anomaly import emerging_issue_score
from src.analytics.complaints import kpi_summary, top_categories
from src.config import (
    AI_OPPORTUNITIES_PATH,
    ARTIFACT_DIR,
    COMPLAINTS_PARQUET,
    DOCS_DIR,
    METADATA_PATH,
    MODEL_REPORT_PATH,
    QUALITY_PROFILE_PATH,
    RAG_EVAL_METRICS_PATH,
    RAG_EVAL_PATH,
    RAG_INDEX_PATH,
    ROOT_DIR,
    TOPIC_REGISTRY_PATH,
)
from src.governance.nist import governance_catalog
from src.nlp.rag_eval import evaluate_retrieval
from src.value.model import ValueAssumptions, calculate_value


def _money(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def generate_final_reports() -> dict[str, object]:
    if not COMPLAINTS_PARQUET.exists():
        raise FileNotFoundError(f"Missing processed complaint store: {COMPLAINTS_PARQUET}")

    complaints = pd.read_parquet(COMPLAINTS_PARQUET)
    metadata = _load_json(METADATA_PATH)
    model = _load_json(MODEL_REPORT_PATH)
    topics = pd.read_csv(TOPIC_REGISTRY_PATH) if TOPIC_REGISTRY_PATH.exists() else pd.DataFrame()
    opportunities = pd.read_csv(AI_OPPORTUNITIES_PATH) if AI_OPPORTUNITIES_PATH.exists() else pd.DataFrame()
    quality = pd.read_csv(QUALITY_PROFILE_PATH) if QUALITY_PROFILE_PATH.exists() else pd.DataFrame()
    governance = governance_catalog(opportunities) if not opportunities.empty else pd.DataFrame()
    rag_report = evaluate_retrieval(complaints, ROOT_DIR / "evaluation" / "rag_eval_set.csv", RAG_EVAL_PATH)
    rag_metrics = _load_json(RAG_EVAL_METRICS_PATH)
    rag_index_metrics = _load_json(RAG_INDEX_PATH.with_suffix(".json"))

    kpis = kpi_summary(complaints)
    top_products = top_categories(complaints, "product", 8)
    top_issues = top_categories(complaints, "issue", 8)
    top_companies = top_categories(complaints, "company", 10)
    emerging = emerging_issue_score(complaints, "issue").head(10)

    baseline_value = calculate_value(ValueAssumptions(observed_complaints=int(kpis["complaints"]), hourly_wage=24.0))
    credit_reporting_volume = int(complaints[complaints["product"].astype(str).str.contains("Credit reporting", case=False, na=False)].shape[0])

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    top_products.to_csv(ARTIFACT_DIR / "final_top_products.csv", index=False)
    top_issues.to_csv(ARTIFACT_DIR / "final_top_issues.csv", index=False)
    top_companies.to_csv(ARTIFACT_DIR / "final_top_companies.csv", index=False)
    emerging.to_csv(ARTIFACT_DIR / "final_emerging_issues.csv", index=False)
    governance.to_csv(ARTIFACT_DIR / "final_governance_catalog.csv", index=False)

    final_metrics = {
        "record_count": int(kpis["complaints"]),
        "date_min": kpis["date_min"],
        "date_max": kpis["date_max"],
        "companies": int(kpis["companies"]),
        "products": int(kpis["products"]),
        "issues": int(kpis["issues"]),
        "states": int(complaints["state"].nunique(dropna=True)),
        "narratives": int(kpis["narratives"]),
        "narrative_share": round(int(kpis["narratives"]) / max(int(kpis["complaints"]), 1), 4),
        "parquet_size_mb": round(COMPLAINTS_PARQUET.stat().st_size / 1_048_576, 2),
        "pipeline_runtime_seconds": metadata.get("runtime_seconds"),
        "limited_extract": metadata.get("limited_extract"),
        "topics": int(len(topics)),
        "topic_narrative_volume": int(topics["volume"].sum()) if "volume" in topics else 0,
        "opportunities": int(len(opportunities)),
        "governance_use_cases": int(governance["ai_use_case"].nunique()) if not governance.empty else 0,
        "governance_controls": int(governance["recommended_controls"].str.count(r"\|").add(1).sum()) if not governance.empty else 0,
        "rag_eval_questions": int(rag_metrics.get("question_count", len(rag_report))),
        "rag_indexed_narratives": rag_index_metrics.get("indexed_narratives"),
        "rag_recall_at_5": rag_metrics.get("recall_at_5"),
        "rag_recall_at_10": rag_metrics.get("recall_at_10"),
        "rag_citation_validity_rate": rag_metrics.get("citation_validity_rate"),
        "rag_metadata_filter_correctness": rag_metrics.get("metadata_filter_correctness"),
        "rag_answer_citation_presence_rate": rag_metrics.get("answer_citation_presence_rate"),
        "rag_unsupported_claim_rate": rag_metrics.get("unsupported_claim_rate"),
        "rag_groundedness_pass_rate": rag_metrics.get("citation_validity_rate", round(float(rag_report["groundedness_gate"].mean()), 3) if not rag_report.empty else None),
        "routing_model_records": model.get("records"),
        "routing_model_classes": model.get("classes"),
        "baseline_macro_f1": model.get("baseline_macro_f1"),
        "model_macro_f1": model.get("model_macro_f1"),
        "credit_reporting_volume": credit_reporting_volume,
        "baseline_estimated_annual_capacity_value": baseline_value["estimated_annual_capacity_value"],
        "baseline_three_year_net_value": baseline_value["three_year_net_value"],
    }
    (ARTIFACT_DIR / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    top_product_line = top_products.iloc[0]
    top_issue_line = top_issues.iloc[0]
    top_company_line = top_companies.iloc[0]
    top_emerging_line = emerging.iloc[0] if not emerging.empty else None

    docs = {
        DOCS_DIR / "DATA_BENCHMARKS.md": _data_benchmarks(final_metrics, metadata),
        DOCS_DIR / "EXECUTIVE_RECOMMENDATION.md": _executive_recommendation(
            final_metrics,
            top_product_line,
            top_issue_line,
            top_company_line,
            top_emerging_line,
            top_products,
            top_issues,
            opportunities,
            baseline_value,
        ),
        DOCS_DIR / "RESUME_EVIDENCE.md": _resume_evidence(final_metrics),
        DOCS_DIR / "business" / "AI_USE_CASE_CATALOG.md": _ai_catalog(opportunities, governance),
    }
    for path, text in docs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    return final_metrics


def _data_benchmarks(metrics: dict[str, object], metadata: dict) -> str:
    return f"""# Data Benchmarks

Generated from the local official-source pipeline.

| Metric | Measured Value |
|---|---:|
| CFPB records processed | {metrics['record_count']:,} |
| Date coverage | {metrics['date_min']} to {metrics['date_max']} |
| Public narratives | {metrics['narratives']:,} |
| Narrative share | {metrics['narrative_share']:.1%} |
| Companies | {metrics['companies']:,} |
| Products | {metrics['products']:,} |
| Issues | {metrics['issues']:,} |
| States / territories | {metrics['states']:,} |
| Compressed Parquet size | {metrics['parquet_size_mb']} MB |
| Pipeline runtime | {metrics['pipeline_runtime_seconds']} seconds |
| NLP topics built | {metrics['topics']:,} |
| Topic-assigned narratives | {metrics['topic_narrative_volume']:,} |
| AI opportunities identified | {metrics['opportunities']:,} |
| RAG evaluation questions | {metrics['rag_eval_questions']:,} |
| RAG indexed narratives | {metrics.get('rag_indexed_narratives') or 0:,} |
| RAG Recall@5 | {(metrics.get('rag_recall_at_5') or 0):.1%} |
| RAG Recall@10 | {(metrics.get('rag_recall_at_10') or 0):.1%} |
| RAG citation validity | {(metrics.get('rag_citation_validity_rate') or 0):.1%} |
| Routing model baseline macro F1 | {metrics['baseline_macro_f1']:.3f} |
| Routing model macro F1 | {metrics['model_macro_f1']:.3f} |

## Source Metadata

```json
{json.dumps(metadata, indent=2)}
```

## Interpretation

This run uses a capped official CFPB extract (`limited_extract=true`) for practical local execution. The engineering path supports full bulk refresh through the same command with the cap removed.
"""


def _executive_recommendation(
    metrics: dict[str, object],
    top_product,
    top_issue,
    top_company,
    top_emerging,
    top_products: pd.DataFrame,
    top_issues: pd.DataFrame,
    opportunities: pd.DataFrame,
    baseline_value: dict,
) -> str:
    emerging_text = "No latest-month emerging issue could be scored." if top_emerging is None else (
        f"`{top_emerging.iloc[0]}` is the highest latest-month emerging-issue signal in this extract, "
        f"with {int(top_emerging['complaints']):,} complaints and a z-score of {top_emerging['z_score']:.2f}."
    )
    opp_lines = "\n".join(
        f"- {row.opportunity_name}: {row.observed_problem} ({int(row.supporting_data):,} supporting complaints)."
        for row in opportunities.head(6).itertuples()
    )
    product_lines = "\n".join(f"- {r.product}: {int(r.complaints):,} complaints ({r.share:.1%})." for r in top_products.itertuples())
    issue_lines = "\n".join(f"- {r.issue}: {int(r.complaints):,} complaints ({r.share:.1%})." for r in top_issues.head(6).itertuples())
    return f"""# Executive Recommendation

## Executive Summary
The platform processed {metrics['record_count']:,} real CFPB complaint records covering {metrics['date_min']} to {metrics['date_max']}. The leading product category in this extract is `{top_product['product']}` with {int(top_product['complaints']):,} complaints. The leading issue is `{top_issue['issue']}` with {int(top_issue['complaints']):,} complaints.

These findings come from official CFPB data and local analysis files. Because this run is capped, read it as a strong project extract, not a final report on the whole market.

## Observed Consumer Problems
{product_lines}

Top issues:
{issue_lines}

## Emerging Risks
{emerging_text}

The emerging-risk monitor uses a trailing-baseline z-score. A spike is a statistical investigation signal, not a causal conclusion.

## Company / Product Insight
`{top_company['company']}` has the highest raw complaint volume in this selected extract with {int(top_company['complaints']):,} complaints. The app deliberately avoids saying this means the company is worse; raw complaint volume must be interpreted with scale, product mix, customer base, and reporting limitations.

## AI Opportunity Areas
{opp_lines}

## Estimated Value Under Explicit Assumptions
Using the baseline scenario across the selected {metrics['record_count']:,} observed complaints:
- Estimated annual capacity value: {_money(baseline_value['estimated_annual_capacity_value'])}.
- Estimated 3-year net value: {_money(baseline_value['three_year_net_value'])}.
- Estimated hours released: {baseline_value['expected_hours_released']:,.0f}.

These are capacity-value estimates, not realized savings. The app exposes handling time, addressable share, time reduction, adoption, wage benchmark, implementation cost, operating cost, success probability, and discount rate.

## Governance Requirements
The highest-priority controls are source-grounded generation, human review for customer-impacting decisions, model evaluation by class, drift monitoring, audit logging, privacy controls, and low-confidence escalation. NIST AI RMF concepts inform the structure, but this project does not claim compliance certification.

## Recommended Priorities
1. Use the app to monitor high-volume credit-reporting and investigation-related complaint categories.
2. Pilot human-in-the-loop complaint routing or agent-assist retrieval before any autonomous customer-impacting workflow.
3. Use topic and retrieval evidence to validate repeated failure modes with operations and compliance SMEs.
4. Run SEC and BLS enrichment for stronger normalized benchmarking and value-model calibration.

## 90-Day Actions
- Run full CFPB ingestion without the record cap.
- Review the topic registry for duplicate or template-like complaint language.
- Expand entity-resolution mappings only where a public registrant match is defensible.
- Define a governed pilot for one low-risk AI support workflow.

## 6-12 Month Roadmap
- Add internal complaint resolution outcomes and QA labels in a real enterprise deployment.
- Replace public-only denominators with customer/account/transaction exposure where available.
- Move from baseline NLP to embeddings or BERTopic after measuring topic quality gains.
- Establish production monitoring, review workflows, and governance sign-off gates.
"""


def _resume_evidence(metrics: dict[str, object]) -> str:
    return f"""# Resume Evidence

Measured after the local official-source pipeline run.

## Verifiable Facts
- Processed {metrics['record_count']:,} real CFPB complaint records.
- Covered {metrics['date_min']} to {metrics['date_max']}.
- Analyzed {metrics['companies']:,} companies, {metrics['products']:,} products, {metrics['issues']:,} issues, and {metrics['states']:,} states/territories.
- Processed {metrics['narratives']:,} public complaint narratives.
- Built {metrics['topics']:,} NLP topics across {metrics['topic_narrative_volume']:,} topic-assigned narratives.
- Identified {metrics['opportunities']:,} observed-pattern AI opportunity rows.
- Created governance coverage for {metrics['governance_use_cases']:,} AI use-case types with {metrics['governance_controls']:,} controls.
- Indexed {metrics.get('rag_indexed_narratives') or 0:,} public narratives for grounded RAG retrieval.
- Evaluated {metrics['rag_eval_questions']:,} RAG questions with {(metrics.get('rag_recall_at_5') or 0):.1%} Recall@5 and {(metrics.get('rag_citation_validity_rate') or 0):.1%} citation validity.
- Trained a narrative routing classifier on {metrics['routing_model_records']:,} narratives across {metrics['routing_model_classes']:,} product classes.
- Improved macro F1 from {metrics['baseline_macro_f1']:.3f} baseline to {metrics['model_macro_f1']:.3f}.

## Safe Wording
- Built a complaint intelligence app using {metrics['record_count']:,} real CFPB records and {metrics.get('rag_indexed_narratives') or 0:,} indexed public narratives.
- Built a grounded RAG layer with hybrid retrieval, metadata filters, complaint citations, and deterministic analytics context.
- Evaluated RAG retrieval on {metrics['rag_eval_questions']:,} analyst questions, measuring {(metrics.get('rag_recall_at_5') or 0):.1%} Recall@5 and {(metrics.get('rag_citation_validity_rate') or 0):.1%} citation validity.
- Trained a TF-IDF + Logistic Regression routing model and improved macro F1 from {metrics['baseline_macro_f1']:.3f} baseline to {metrics['model_macro_f1']:.3f}.

## Wording To Avoid
- Do not say the project analyzed millions of complaints unless the full pipeline is rerun and verified.
- Do not say SEC, BLS, or FRED are integrated in this release.
- Do not say the model found realized savings.
- Do not say the RAG assistant proves misconduct.
- Do not say the LLM calculated business metrics.

## Current Best Resume Bullet Drafts
- Built a CFPB complaint intelligence app analyzing {metrics['record_count']:,} real financial-services complaints, using trend analytics, topic modeling, routing classification, and grounded RAG to surface operational risk themes with cited evidence.
- Developed a grounded analyst-assistant layer over {metrics.get('rag_indexed_narratives') or 0:,} public complaint narratives with hybrid retrieval, metadata filters, deterministic analytics context, and {metrics['rag_eval_questions']:,}-question RAG evaluation achieving {(metrics.get('rag_recall_at_5') or 0):.1%} Recall@5 and {(metrics.get('rag_citation_validity_rate') or 0):.1%} citation validity.

## Usage Note
These bullets are based on the capped local official extract. If the full CFPB dataset is run later, regenerate this file before using larger numbers.
"""


def _ai_catalog(opportunities: pd.DataFrame, governance: pd.DataFrame) -> str:
    rows = []
    for _, opp in opportunities.head(20).iterrows():
        controls = ""
        if not governance.empty:
            match = governance[governance["ai_use_case"] == opp["opportunity_name"]]
            if not match.empty:
                controls = match.iloc[0]["recommended_controls"]
        rows.append(
            f"""## {opp['opportunity_name']}
- Observed problem: {opp['observed_problem']}.
- Supporting data: {int(opp['supporting_data']):,} complaints.
- Affected product: {opp['affected_products']}.
- Affected issue: {opp['affected_issues']}.
- Proposed intervention: {opp['potential_ai_intervention']}.
- Human role: {opp['expected_human_role']}.
- Data requirements: {opp['data_requirements']}.
- Risk considerations: {opp['risk_considerations']}.
- Controls: {controls}
"""
        )
    return "# AI Use Case Catalog\n\nBuilt from observed CFPB complaint patterns in the local analytical run.\n\n" + "\n".join(rows)


def main() -> None:
    metrics = generate_final_reports()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
