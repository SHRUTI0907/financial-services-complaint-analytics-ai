from __future__ import annotations

import pandas as pd


AI_PATTERNS = [
    ("Complaint routing and triage", ["routing", "wrong department", "response", "delay", "timely"], "Route complaints to the right queue and prioritize escalations."),
    ("Agent-assist retrieval", ["documentation", "information", "explain", "statement", "account"], "Retrieve policy, account, and prior-case context for human agents."),
    ("Root-cause detection", ["fee", "payment", "fraud", "reporting", "servicing"], "Detect repeated failure modes across products and companies."),
    ("Complaint summarization", ["narrative", "documents", "letter", "call", "email"], "Summarize long complaint histories with citations for reviewer validation."),
    ("Emerging-risk monitoring", ["unauthorized", "fraud", "closed", "blocked", "identity"], "Alert risk teams when themes exceed statistical baselines."),
]


def derive_ai_opportunities(complaints: pd.DataFrame, emerging: pd.DataFrame | None = None) -> pd.DataFrame:
    issue_counts = complaints.groupby(["product", "issue"], dropna=False).size().reset_index(name="observed_complaints")
    issue_counts = issue_counts.sort_values("observed_complaints", ascending=False).head(30)
    rows = []
    for _, row in issue_counts.iterrows():
        text = f"{row['product']} {row['issue']}".lower()
        matched = [pattern for pattern in AI_PATTERNS if any(keyword in text for keyword in pattern[1])]
        if not matched:
            matched = [("Operational intelligence workbench", ["volume"], "Use analytics to identify repeated complaint friction and escalation patterns.")]
        for name, _, intervention in matched[:1]:
            rows.append(
                {
                    "opportunity_name": name,
                    "observed_problem": f"High complaint volume for {row['product']} / {row['issue']}",
                    "supporting_data": int(row["observed_complaints"]),
                    "affected_products": row["product"],
                    "affected_issues": row["issue"],
                    "potential_ai_intervention": intervention,
                    "expected_human_role": "Human reviewer remains accountable for customer-impacting decisions.",
                    "data_requirements": "Complaint text, category history, response outcomes, and quality-reviewed resolution labels.",
                    "risk_considerations": "Misrouting, hallucinated summaries, privacy leakage, unequal model performance, and overreliance.",
                    "evidence_type": "Observed CFPB complaint pattern; intervention is proposed, not observed.",
                }
            )
    result = pd.DataFrame(rows).drop_duplicates(subset=["opportunity_name", "affected_products", "affected_issues"])
    if emerging is not None and not emerging.empty and "emerging_issue_score" in emerging.columns:
        result["emerging_issue_signal"] = result["affected_issues"].map(dict(zip(emerging.iloc[:, 0], emerging["emerging_issue_score"]))).fillna(0)
    return result
