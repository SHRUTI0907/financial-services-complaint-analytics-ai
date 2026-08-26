from __future__ import annotations

import json
import os
from dataclasses import asdict
from urllib.request import Request, urlopen

import pandas as pd

from src.config import COMPLAINTS_PARQUET, RAG_INDEX_PATH
from src.rag.analytics_tools import run_default_analytics
from src.rag.retrieval import RagFilters, hybrid_retrieve

SYSTEM_PROMPT = """You are an AI Complaint Intelligence Analyst for CFPB complaint evidence.
Use only the supplied evidence and analytics context.
Do not invent metrics, complaint IDs, companies, dates, or causal explanations.
Use the structure: Finding, Evidence, Why it matters, Caveat.
Every factual claim needs either a CFPB complaint citation or an Analytics citation.
If evidence is weak or missing, say so directly.
Do not infer misconduct. Do not say a company is better or worse without denominator caveats.
Clearly label estimates and scenarios."""


def _format_analytics(metrics: list[dict[str, object]]) -> str:
    return "\n".join(json.dumps(metric, default=str) for metric in metrics)


def _format_evidence(evidence: pd.DataFrame, max_chars: int = 700) -> str:
    lines = []
    for row in evidence.head(8).to_dict("records"):
        text = str(row.get("clean_narrative", ""))[:max_chars]
        lines.append(
            f"{row.get('citation')} date={row.get('date_received')} company={row.get('company')} "
            f"product={row.get('product')} issue={row.get('issue')} text={text}"
        )
    return "\n".join(lines)


def build_context(question: str, complaints: pd.DataFrame, filters: RagFilters | None = None, top_k: int = 8) -> dict[str, object]:
    filters = filters or RagFilters()
    retrieval = hybrid_retrieve(question, filters=filters, top_k=top_k, index_path=RAG_INDEX_PATH)
    analytics = run_default_analytics(complaints, filters)
    return {
        "question": question,
        "filters": asdict(filters),
        "analytics": analytics,
        "evidence": retrieval.evidence,
        "trace": {
            "parsed_intent": "general_analytical_question",
            "filters": asdict(filters),
            "analytics_calls": [metric["metric_id"] for metric in analytics],
            "retrieval": retrieval.trace,
        },
    }


def deterministic_answer(context: dict[str, object]) -> str:
    evidence: pd.DataFrame = context["evidence"]
    analytics: list[dict[str, object]] = context["analytics"]
    volume = next((m for m in analytics if m["metric_id"] == "complaint_volume"), {})
    growth = next((m for m in analytics if m["metric_id"] == "growth_rate"), {})
    top_issues = next((m for m in analytics if m["metric_id"] == "top_issues"), {})
    if evidence.empty:
        return (
            "Finding\n"
            "I do not have enough matching complaint evidence to answer this safely.\n\n"
            "Evidence\n"
            f"The filtered dataset contains {volume.get('complaints', 0):,} complaints [Analytics: complaint volume].\n\n"
            "Why it matters\n"
            "The assistant should not fill gaps with guesses.\n\n"
            "Caveat\n"
            "Try loosening the company, product, issue, state, or date filters."
        )

    issue_bits = []
    for row in top_issues.get("rows", [])[:3]:
        issue_bits.append(f"{row.get('issue')} ({int(row.get('complaints', 0)):,})")
    issue_text = ", ".join(issue_bits) if issue_bits else "not enough issue detail"
    citations = " ".join(evidence["citation"].head(3).tolist())
    growth_text = "growth could not be calculated because there was not enough history"
    if growth.get("status") != "insufficient_history" and growth.get("growth_rate") is not None:
        growth_text = f"recent {growth.get('months')}-month volume changed by {growth.get('growth_rate'):.1%}"
    return (
        "Finding\n"
        f"The filtered complaint set contains {volume.get('complaints', 0):,} records. The leading issues are {issue_text} [Analytics: top issues].\n\n"
        "Evidence\n"
        f"The top retrieved complaint narratives include records such as {citations}. {growth_text} [Analytics: recent complaint growth].\n\n"
        "Why it matters\n"
        "Repeated complaint language and high-volume issues are good places to look for process friction, routing problems, unclear communications, or review bottlenecks.\n\n"
        "Caveat\n"
        "This is complaint evidence, not proof of misconduct or a representative customer survey."
    )


def _call_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    payload = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    req = Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urlopen(req, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _call_anthropic(prompt: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
    payload = {
        "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
        "max_tokens": 900,
        "temperature": 0.1,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
    )
    with urlopen(req, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["content"][0]["text"]


def generate_answer(context: dict[str, object], use_llm: bool = True) -> dict[str, object]:
    evidence: pd.DataFrame = context["evidence"]
    prompt = (
        f"Question: {context['question']}\n\n"
        f"Filters: {json.dumps(context['filters'], default=str)}\n\n"
        f"Analytics context:\n{_format_analytics(context['analytics'])}\n\n"
        f"Complaint evidence:\n{_format_evidence(evidence)}"
    )
    provider = os.getenv("LLM_PROVIDER", "").lower().strip()
    llm_configured = provider in {"openai", "anthropic"} and (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    if use_llm and llm_configured:
        answer = _call_openai(prompt) if provider == "openai" else _call_anthropic(prompt)
        mode = f"llm:{provider}"
    else:
        answer = deterministic_answer(context)
        mode = "deterministic_no_key_fallback"
    return {
        "answer": answer,
        "mode": mode,
        "citations": evidence["citation"].head(8).tolist() if not evidence.empty else [],
        "evidence": evidence,
        "analytics": context["analytics"],
        "prompt_context": prompt,
        "trace": context["trace"],
    }


def answer_question(question: str, filters: RagFilters | None = None, top_k: int = 8, use_llm: bool = True) -> dict[str, object]:
    complaints = pd.read_parquet(COMPLAINTS_PARQUET)
    context = build_context(question, complaints, filters, top_k)
    result = generate_answer(context, use_llm=use_llm)
    return result
