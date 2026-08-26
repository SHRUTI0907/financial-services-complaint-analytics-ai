from __future__ import annotations

import pandas as pd

TRUSTWORTHY_DIMENSIONS = [
    "validity_reliability",
    "safety",
    "security_resilience",
    "accountability_transparency",
    "explainability_interpretability",
    "privacy",
    "fairness_harmful_bias",
    "human_oversight",
]

CONTROL_LIBRARY = {
    "Complaint routing and triage": [
        "Measure precision, recall, and macro F1 by product and issue before deployment.",
        "Keep manual escalation available for low-confidence routing.",
        "Monitor drift in product and issue distribution monthly.",
    ],
    "Complaint summarization": [
        "Require source-grounded summaries with complaint-ID citations.",
        "Use human review for external or customer-impacting summaries.",
        "Log prompts, retrieved evidence, generated summaries, and reviewer corrections.",
    ],
    "Agent-assist retrieval": [
        "Restrict retrieval to approved knowledge bases and complaint records users are permitted to access.",
        "Show source passages and confidence cues to agents.",
        "Audit retrieval failures and unsupported answer attempts.",
    ],
    "Root-cause detection": [
        "Validate clusters with operations SMEs before action.",
        "Track false positives and missed recurring issues.",
        "Document taxonomy changes that may alter trend interpretation.",
    ],
    "Emerging-risk monitoring": [
        "Use transparent thresholds and alert review queues.",
        "Separate statistical anomaly from causal explanation.",
        "Review alerts with risk, legal, operations, and product owners.",
    ],
    "Executive RAG assistant": [
        "Route all metrics through deterministic analytics APIs.",
        "Require citations to complaint IDs, topic IDs, or generated metric tables.",
        "Use abstention when evidence is unavailable.",
    ],
}


def risk_tier(use_case_name: str, customer_impact: str = "medium", uses_genai: bool = True) -> str:
    score = 1
    if customer_impact == "high":
        score += 2
    elif customer_impact == "medium":
        score += 1
    if uses_genai:
        score += 1
    if any(term in use_case_name.lower() for term in ["routing", "summarization", "assistant"]):
        score += 1
    if score >= 5:
        return "High"
    if score >= 3:
        return "Moderate"
    return "Low"


def governance_catalog(opportunities: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, opp in opportunities.iterrows():
        name = opp["opportunity_name"]
        tier = risk_tier(name)
        rows.append(
            {
                "ai_use_case": name,
                "intended_purpose": opp["potential_ai_intervention"],
                "affected_stakeholders": "Consumers, complaint operations agents, risk managers, compliance reviewers, executives.",
                "data_used": "Public CFPB complaint metadata and narratives; internal deployment would require permissioned complaint and resolution data.",
                "decision_impact": "Operational prioritization and employee decision support; final customer-impacting action remains human-owned.",
                "risk_tier": tier,
                "nist_dimensions": ", ".join(TRUSTWORTHY_DIMENSIONS),
                "major_risks": opp["risk_considerations"],
                "human_in_the_loop": "Required for customer-impacting decisions and unresolved low-confidence outputs.",
                "testing_requirements": "Offline evaluation, SME review, segmented performance checks, drift monitoring, and adversarial prompt testing where GenAI is used.",
                "monitoring_requirements": "Volume drift, model confidence, appeal/escalation rates, citation coverage, unresolved cases, and reviewer override rate.",
                "escalation_rules": "Escalate low confidence, missing evidence, regulatory-sensitive categories, suspected fraud, or consumer harm signals.",
                "recommended_controls": " | ".join(CONTROL_LIBRARY.get(name, CONTROL_LIBRARY["Root-cause detection"])),
                "methodology_note": "NIST AI RMF concepts inform dimensions; this is a project-specific control recommendation, not a compliance certification.",
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["ai_use_case"])
