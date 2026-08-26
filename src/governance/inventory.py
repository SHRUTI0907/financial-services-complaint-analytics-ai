from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import AI_RISK_REGISTER_PATH, AI_SYSTEM_INVENTORY_PATH, GOVERNANCE_CONTROLS_PATH


def ai_system_inventory() -> pd.DataFrame:
    rows = [
        {
            "system_name": "Complaint-routing classifier",
            "purpose": "Predict CFPB product category from public complaint narrative text for triage analysis.",
            "users": "Analysts, operations leads",
            "inputs": "Public complaint narratives",
            "outputs": "Predicted product class and evaluation metrics",
            "decision_impact": "Decision support only; no customer-impacting automation.",
            "model_method": "TF-IDF + Logistic Regression",
            "human_involvement": "Analyst reviews model performance and errors before use.",
            "failure_modes": "Misclassification, class imbalance, stale taxonomy, poor minority-class recall.",
            "affected_stakeholders": "Consumers, complaint operations teams, risk/compliance reviewers",
            "data_risks": "Narrative availability bias, redaction artifacts, category drift.",
            "operational_risks": "Automation bias, wrong queue prioritization, overconfidence.",
            "monitoring_metrics": "macro F1, per-class precision/recall, confusion matrix, class distribution drift",
        },
        {
            "system_name": "NMF topic model",
            "purpose": "Group recurring narrative language into interpretable themes.",
            "users": "Analysts, product/risk teams",
            "inputs": "Cleaned public complaint narratives",
            "outputs": "Topic labels, terms, topic volumes",
            "decision_impact": "Investigation prioritization.",
            "model_method": "TF-IDF + NMF",
            "human_involvement": "Analyst names and reviews topics.",
            "failure_modes": "Topic instability, duplicated themes, template-language bias.",
            "affected_stakeholders": "Analysts, operations teams, consumers indirectly",
            "data_risks": "Sparse narratives and repeated complaint templates.",
            "operational_risks": "Misreading topic clusters as root cause.",
            "monitoring_metrics": "topic volume, top terms, example narrative review, topic drift",
        },
        {
            "system_name": "Emerging-risk detector",
            "purpose": "Flag unusual recent complaint volume versus trailing baseline.",
            "users": "Risk analysts, operations leads",
            "inputs": "Complaint dates and selected grouping fields",
            "outputs": "Z-score spike signals",
            "decision_impact": "Investigation lead only.",
            "model_method": "Trailing baseline z-score",
            "human_involvement": "Human validates alerts and business context.",
            "failure_modes": "False positives, seasonality, taxonomy changes, low-volume noise.",
            "affected_stakeholders": "Risk/compliance teams, consumers indirectly",
            "data_risks": "Complaint reporting lag and public data incompleteness.",
            "operational_risks": "Treating correlation as cause.",
            "monitoring_metrics": "alert count, false-positive review rate, stale data age",
        },
        {
            "system_name": "Hybrid retrieval",
            "purpose": "Retrieve CFPB complaint evidence for analyst questions.",
            "users": "Analysts, executives reviewing evidence",
            "inputs": "Question text, metadata filters, indexed public narratives",
            "outputs": "Ranked complaint evidence with citations",
            "decision_impact": "Evidence discovery.",
            "model_method": "TF-IDF lexical + local LSA dense retrieval + rank fusion",
            "human_involvement": "User reviews cited complaint rows.",
            "failure_modes": "Retrieval miss, weak semantic match, filter mistake, stale index.",
            "affected_stakeholders": "Analysts, executives, operations teams",
            "data_risks": "Only public narratives are searchable.",
            "operational_risks": "Overreliance on top retrieved rows.",
            "monitoring_metrics": "Recall@5, Recall@10, citation validity, metadata-filter correctness",
        },
        {
            "system_name": "Grounded LLM answer layer",
            "purpose": "Draft analyst/executive answers from retrieved evidence and deterministic analytics.",
            "users": "Analysts, executives",
            "inputs": "Question, retrieved complaint evidence, deterministic analytics context",
            "outputs": "Cited structured answer",
            "decision_impact": "Narrative decision support only.",
            "model_method": "Optional OpenAI/Anthropic provider or deterministic no-key fallback",
            "human_involvement": "Human reviews all generated answers before use.",
            "failure_modes": "Hallucination, unsupported claim, bad citation, prompt injection, provider outage.",
            "affected_stakeholders": "Analysts, executives, consumers indirectly",
            "data_risks": "Sensitive public complaint text, prompt leakage, stale context.",
            "operational_risks": "Automation bias and misplaced trust in generated text.",
            "monitoring_metrics": "citation presence, unsupported-claim rate, abstention rate, retrieval quality",
        },
        {
            "system_name": "AI value/opportunity model",
            "purpose": "Estimate capacity value under explicit assumptions for AI-assisted complaint workflows.",
            "users": "Business analysts, product managers, finance partners",
            "inputs": "Observed complaint volume, external wage benchmark when available, user assumptions",
            "outputs": "Hours released, capacity value, NPV, ROI, payback, scenario and Monte Carlo results",
            "decision_impact": "Business-case prioritization.",
            "model_method": "Deterministic financial model + seeded Monte Carlo",
            "human_involvement": "Business owner validates assumptions.",
            "failure_modes": "Bad assumptions, benchmark mismatch, treating modeled value as realized savings.",
            "affected_stakeholders": "Operations teams, finance, executives",
            "data_risks": "No internal handle-time or actual payroll data in public dataset.",
            "operational_risks": "Overstated ROI, underestimating adoption/change management.",
            "monitoring_metrics": "assumption ranges, NPV distribution, probability NPV > 0, payback probability",
        },
    ]
    return pd.DataFrame(rows)


def risk_register() -> pd.DataFrame:
    risks = [
        ("R001", "Grounded LLM answer layer", "Unsupported or hallucinated executive claim", 3, 5, "Require citations and deterministic metric context", "Analytics owner", "unsupported_claim_rate", "> 0%"),
        ("R002", "Hybrid retrieval", "Relevant complaint evidence is not retrieved", 3, 4, "Evaluate Recall@K and show retrieved rows for review", "Analytics engineer", "recall_at_5", "< 85%"),
        ("R003", "Hybrid retrieval", "Citation points to wrong or missing complaint ID", 2, 5, "Citation validity test on every eval run", "Analytics engineer", "citation_validity_rate", "< 98%"),
        ("R004", "Complaint-routing classifier", "Misclassification sends work to wrong queue", 3, 4, "Monitor macro F1 and per-class recall; keep human override", "Model owner", "macro_f1", "< baseline target"),
        ("R005", "Complaint-routing classifier", "Uneven performance across product classes", 4, 4, "Review confusion matrix and per-class metrics", "Model owner", "min_class_recall", "< 50%"),
        ("R006", "NMF topic model", "Topics become unstable after data refresh", 3, 3, "Compare top terms and example complaints between runs", "Analytics owner", "topic_review_pass_rate", "< 90%"),
        ("R007", "Emerging-risk detector", "Spike is misread as causal proof", 3, 4, "Label z-score as investigation signal only", "Risk owner", "reviewed_alert_rate", "< 100%"),
        ("R008", "AI value/opportunity model", "Modeled capacity value is presented as realized savings", 3, 5, "UI labels observed inputs, assumptions, benchmarks, and model outputs", "Business owner", "labeling_review", "any failure"),
        ("R009", "AI value/opportunity model", "Assumptions produce inflated ROI", 3, 4, "Scenario, sensitivity, and Monte Carlo distributions", "Finance partner", "probability_npv_positive", "low confidence"),
        ("R010", "Grounded LLM answer layer", "Prompt injection changes answer rules", 2, 4, "System prompt, evidence-only context, human review", "Product owner", "prompt_injection_test_pass", "< 100%"),
        ("R011", "All systems", "Public data is stale or incomplete", 4, 3, "Show source dates and retrieval timestamps", "Data owner", "data_age_days", "> refresh SLA"),
        ("R012", "All systems", "Privacy mishandling of public complaint narratives", 2, 5, "Use redacted CFPB public narratives only; avoid exporting unnecessary text", "Data owner", "privacy_review", "any failure"),
        ("R013", "SEC normalization", "Company-to-registrant mapping combines brands incorrectly", 3, 5, "High-confidence gate and manual review artifact", "Data owner", "manual_review_required", "unreviewed normalized metric"),
        ("R014", "SEC normalization", "Assets/revenue denominator is treated as market share", 3, 4, "UI caveat and normalization labels", "Analytics owner", "denominator_caveat_visible", "false"),
    ]
    rows = []
    for risk_id, system, risk, likelihood, impact, control, owner, metric, threshold in risks:
        rows.append(
            {
                "risk_id": risk_id,
                "system": system,
                "risk": risk,
                "likelihood": likelihood,
                "impact": impact,
                "severity": likelihood * impact,
                "control": control,
                "owner": owner,
                "monitoring_metric": metric,
                "escalation_threshold": threshold,
            }
        )
    return pd.DataFrame(rows)


def governance_controls() -> pd.DataFrame:
    rows = [
        ("G001", "Grounded LLM answer layer", "Citation-required answer policy", "Implemented", "Answers cite CFPB complaints or analytics outputs."),
        ("G002", "Grounded LLM answer layer", "Insufficient-evidence abstention", "Implemented", "Fallback abstains when no evidence is retrieved."),
        ("G003", "Hybrid retrieval", "Metadata-filter correctness test", "Implemented", "RAG eval includes impossible filters and expected filters."),
        ("G004", "Hybrid retrieval", "Retrieval quality threshold", "Implemented", "Recall@5 and Recall@10 are calculated."),
        ("G005", "Complaint-routing classifier", "Baseline comparison", "Implemented", "Macro F1 is compared with most-frequent baseline."),
        ("G006", "Complaint-routing classifier", "Per-class evaluation artifact", "Implemented", "Model report stores classification details where generated."),
        ("G007", "NMF topic model", "Human topic review", "Designed", "Topic terms and example evidence are visible for analyst review."),
        ("G008", "Emerging-risk detector", "Causal caveat", "Implemented", "UI labels z-score as investigation signal."),
        ("G009", "AI value/opportunity model", "Assumption transparency", "Implemented", "UI exposes handling time, adoption, addressable share, costs, and success probability."),
        ("G010", "AI value/opportunity model", "Monte Carlo uncertainty", "Implemented", "Seeded simulation produces NPV and payback probabilities."),
        ("G011", "SEC normalization", "Manual-review gate", "Implemented", "Only high-confidence non-review mappings can normalize."),
        ("G012", "All systems", "No compliance-certification claim", "Implemented", "Docs state NIST-informed, not certified."),
    ]
    return pd.DataFrame(rows, columns=["control_id", "system", "control", "status", "evidence"])


def write_governance_artifacts(
    inventory_path: Path = AI_SYSTEM_INVENTORY_PATH,
    risk_path: Path = AI_RISK_REGISTER_PATH,
    controls_path: Path = GOVERNANCE_CONTROLS_PATH,
) -> dict[str, int]:
    inventory = ai_system_inventory()
    risks = risk_register()
    controls = governance_controls()
    inventory.to_csv(inventory_path, index=False)
    risks.to_csv(risk_path, index=False)
    controls.to_csv(controls_path, index=False)
    return {
        "governed_systems": int(len(inventory)),
        "documented_risks": int(len(risks)),
        "implemented_or_designed_controls": int(len(controls)),
        "monitoring_metrics": int(inventory["monitoring_metrics"].str.split(",").explode().str.strip().nunique()),
    }
