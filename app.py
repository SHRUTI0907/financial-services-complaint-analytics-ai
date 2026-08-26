from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics.anomaly import emerging_issue_score
from src.analytics.benchmarking import company_benchmark
from src.analytics.complaints import kpi_summary, top_categories, volume_by_month
from src.config import ARTIFACT_DIR, COMPLAINTS_PARQUET, METADATA_PATH, MODEL_REPORT_PATH, RAG_EVAL_METRICS_PATH, TOPIC_REGISTRY_PATH
from src.rag.assistant import build_context, generate_answer
from src.rag.retrieval import RagFilters


st.set_page_config(page_title="Financial Services Complaint Analytics", layout="wide")

CSS = """
<style>
:root {
  --ink: #171714;
  --soft-ink: #30312c;
  --muted: #666a60;
  --paper: #f8f7f2;
  --panel: #ffffff;
  --panel-2: #f0f1ea;
  --line: #ddd9cc;
  --green: #245f4b;
  --mint: #e4eee8;
  --gold: #a9752a;
  --red: #a14635;
  --blue: #315f80;
  --violet: #574a70;
  --pencil: rgba(23,23,20,.72);
}
.stApp {
  background:
    linear-gradient(rgba(23,23,20,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(23,23,20,.025) 1px, transparent 1px),
    #f8f7f2;
  background-size: 44px 44px, 44px 44px, auto;
  color: var(--ink);
}
[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, rgba(255,255,255,.05), transparent 14rem),
    #1c1d19;
  border-right: 1px solid #33352e;
}
[data-testid="stSidebar"] * { color: #f4efe3 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 8px;
  padding: .38rem .55rem;
  margin-bottom: .22rem;
  background: rgba(255,255,255,.035);
}
.block-container { padding-top: 1.6rem; max-width: 1480px; }
h1, h2, h3 { letter-spacing: 0; color: #161512; }
h1 { font-size: clamp(2rem, 3.4vw, 3.6rem); line-height: 1.02; max-width: 1040px; }
h2 { margin-top: 1.8rem; }
.stTabs [data-baseweb="tab-list"] { gap: .5rem; }
.stTabs [data-baseweb="tab"] {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255,255,255,.72);
  padding: .5rem .8rem;
}
.eyebrow {
  color: var(--green);
  font-weight: 800;
  font-size: .86rem;
  letter-spacing: 0;
}
.page-subtitle {
  color: var(--muted);
  font-size: 1.05rem;
  line-height: 1.55;
  max-width: 920px;
  margin: -.45rem 0 1.2rem 0;
}
.hero-shell {
  border: 1px solid var(--line);
  background: rgba(255,255,255,.72);
  padding: 1.25rem;
  border-radius: 8px;
  box-shadow: 0 18px 50px rgba(39, 48, 36, .08);
  margin-bottom: 1.1rem;
}
.hero-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(300px, .8fr);
  gap: 1rem;
  align-items: stretch;
}
.hero-copy {
  min-height: 310px;
  border-radius: 8px;
  padding: 2rem;
  background:
    linear-gradient(90deg, rgba(36,95,75,.08) 1px, transparent 1px),
    linear-gradient(rgba(36,95,75,.08) 1px, transparent 1px),
    #fbfaf5;
  background-size: 28px 28px;
  color: var(--ink);
  position: relative;
  overflow: hidden;
}
.hero-copy:after {
  content: "field notes";
  position: absolute;
  right: 1.35rem;
  bottom: 1.1rem;
  color: rgba(36,95,75,.22);
  font-weight: 800;
  font-size: 2.4rem;
}
.hero-copy h1, .hero-copy h2, .hero-copy p { color: var(--ink); position: relative; z-index: 1; }
.hero-copy h1 { margin: .45rem 0 .95rem 0; }
.hero-copy p { color: var(--pencil); max-width: 760px; font-size: 1.08rem; line-height: 1.55; }
.hero-kicker {
  color: var(--green);
  font-size: .88rem;
  font-weight: 900;
  letter-spacing: 0;
}
.hero-rail {
  display: grid;
  gap: .75rem;
}
.rail-card, .source-tile, .command-card, .method-card {
  background: rgba(255,255,255,.88);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 1rem;
  box-shadow: 0 10px 28px rgba(39,48,36,.06);
}
.rail-card strong, .source-tile strong, .command-card strong, .method-card strong {
  display: block;
  color: var(--ink);
  font-size: .92rem;
  margin-bottom: .3rem;
}
.rail-card span, .source-tile span, .command-card span, .method-card span {
  color: var(--muted);
  line-height: 1.45;
  font-size: .88rem;
}
.mosaic {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: .8rem;
  margin: 1rem 0;
}
.mosaic .source-tile:nth-child(1) { border-top: 4px solid var(--green); }
.mosaic .source-tile:nth-child(2) { border-top: 4px solid var(--blue); }
.mosaic .source-tile:nth-child(3) { border-top: 4px solid var(--gold); }
.mosaic .source-tile:nth-child(4) { border-top: 4px solid var(--violet); }
.command-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: .8rem;
  margin: 1rem 0 1.25rem;
}
.signal-strip {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: .7rem;
  margin: 1rem 0 1.2rem;
}
.signal {
  border-radius: 8px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,.78);
  padding: .85rem;
  min-height: 92px;
}
.signal strong {
  display: block;
  color: var(--ink);
  font-size: 1rem;
  margin-bottom: .28rem;
}
.signal span {
  color: var(--muted);
  font-size: .78rem;
  line-height: 1.35;
}
.story-board {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: .85rem;
  margin: 1rem 0;
}
.story-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255,255,255,.88);
  padding: 1.05rem;
  min-height: 150px;
  box-shadow: 0 10px 28px rgba(39,48,36,.06);
}
.story-card small {
  display: block;
  color: var(--green);
  font-weight: 850;
  margin-bottom: .45rem;
}
.story-card strong {
  display: block;
  font-size: 1.02rem;
  margin-bottom: .4rem;
}
.story-card p {
  color: var(--muted);
  line-height: 1.52;
  margin: 0;
}
.talk-track {
  border-left: 4px solid var(--green);
  background: rgba(255,255,255,.82);
  padding: 1rem 1.1rem;
  border-radius: 6px;
  color: var(--soft-ink);
  line-height: 1.58;
  margin: 1rem 0;
}
.metric-card {
  background: rgba(255,255,255,.9);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 1.05rem 1rem;
  min-height: 126px;
  box-shadow: 0 10px 28px rgba(40,48,35,.06);
  position: relative;
  overflow: hidden;
}
.metric-card:before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  right: 0;
  height: 4px;
  background: var(--green);
}
.metric-card .label { color: var(--muted); font-size: .82rem; font-weight: 800; }
.metric-card .value { color: var(--ink); font-size: 1.78rem; font-weight: 850; margin-top: .45rem; }
.metric-card .note { color: var(--muted); font-size: .83rem; margin-top: .35rem; }
.warning {
  border-left: 4px solid var(--gold);
  background: rgba(255,255,255,.72);
  padding: .85rem 1rem;
  border-radius: 6px;
  color: #3d3324;
  box-shadow: 0 10px 28px rgba(40,48,35,.06);
  margin: .75rem 0 1rem;
}
.source-pill {
  display: inline-block;
  border: 1px solid var(--line);
  padding: .28rem .55rem;
  border-radius: 99px;
  margin-right: .35rem;
  color: var(--muted);
  background: rgba(255,255,255,.7);
}
.stDataFrame, [data-testid="stDataFrame"] {
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--line);
  box-shadow: 0 12px 36px rgba(39,48,36,.06);
}
div[data-testid="stMetricValue"] { color: var(--ink); }
@media (max-width: 900px) {
  .hero-grid, .mosaic, .command-grid, .signal-strip, .story-board { grid-template-columns: 1fr; }
  .hero-copy { min-height: 320px; padding: 1.35rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLOT_COLORS = ["#1f5f46", "#2f5f88", "#b9852d", "#9e3f3a", "#554a7a", "#6f7f4f", "#b65f45"]


@st.cache_data(show_spinner=False)
def load_complaints() -> pd.DataFrame | None:
    if not COMPLAINTS_PARQUET.exists():
        return None
    return pd.read_parquet(COMPLAINTS_PARQUET)


def styled_fig(fig, height: int = 390):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.78)",
        colorway=PLOT_COLORS,
        font={"family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", "color": "#2a2925"},
        title={"font": {"size": 18, "color": "#12120f"}, "x": 0.02, "xanchor": "left"},
        margin={"l": 18, "r": 18, "t": 72, "b": 82},
        legend={"orientation": "h", "yanchor": "top", "y": -0.16, "xanchor": "left", "x": 0},
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(103,107,97,.16)", zeroline=False, title_font={"size": 12})
    fig.update_yaxes(showgrid=True, gridcolor="rgba(103,107,97,.16)", zeroline=False, title_font={"size": 12})
    return fig


def page_header(kicker: str, title: str, body: str = "") -> None:
    st.markdown(f"<div class='eyebrow'>{kicker}</div>", unsafe_allow_html=True)
    st.title(title)
    if body:
        st.markdown(f"<div class='page-subtitle'>{body}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"<div class='metric-card'><div class='label'>{label}</div><div class='value'>{value}</div><div class='note'>{note}</div></div>",
        unsafe_allow_html=True,
    )


def load_artifact_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_artifact_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def signal_strip(items: list[tuple[str, str]]) -> None:
    html = "".join(f"<div class='signal'><strong>{value}</strong><span>{label}</span></div>" for value, label in items)
    st.markdown(f"<div class='signal-strip'>{html}</div>", unsafe_allow_html=True)


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Filters")
    products = st.sidebar.multiselect("Product", sorted(df["product"].dropna().unique()), default=[])
    states = st.sidebar.multiselect("State", sorted(df["state"].dropna().unique()), default=[])
    min_date, max_date = pd.to_datetime(df["date_received"]).min(), pd.to_datetime(df["date_received"]).max()
    date_range = st.sidebar.date_input("Date received", value=(min_date.date(), max_date.date()))
    out = df.copy()
    if products:
        out = out[out["product"].isin(products)]
    if states:
        out = out[out["state"].isin(states)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        out = out[(pd.to_datetime(out["date_received"]) >= pd.Timestamp(date_range[0])) & (pd.to_datetime(out["date_received"]) <= pd.Timestamp(date_range[1]))]
    return out


def empty_state() -> None:
    st.error("Processed CFPB data artifact not found.")


df = load_complaints()
if df is None:
    empty_state()
    st.stop()

filtered = sidebar_filters(df)
final_metrics = load_artifact_json(ARTIFACT_DIR / "final_metrics.json")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Complaint Patterns",
        "Company View",
        "Narrative Search",
        "Model Performance",
    ],
)

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}

if page == "Overview":
    page_header(
        "Overview",
        "Financial Services Complaint Analytics & AI Decision Support Platform",
    )
    kpi = kpi_summary(filtered)
    signal_strip(
        [
            (f"{kpi['complaints']:,}", "CFPB complaints analyzed"),
            (f"{kpi['companies']:,}", "Companies"),
            (f"{kpi['products']:,}", "Products"),
            (f"{kpi['issues']:,}", "Issues"),
            (f"{kpi['narratives']:,}", "Public narratives"),
        ]
    )
    left, right = st.columns([1.45, 1])
    with left:
        monthly = volume_by_month(filtered)
        st.plotly_chart(styled_fig(px.line(monthly, x="month", y="complaints", title="Complaint Volume Over Time"), 430), width="stretch")
    with right:
        st.plotly_chart(styled_fig(px.bar(top_categories(filtered, "product", 10), x="complaints", y="product", orientation="h", title="Top Products"), 430), width="stretch")

elif page == "Complaint Patterns":
    page_header(
        "Analytics",
        "Products, issues, trends, and anomaly signals",
    )
    dimension = st.selectbox("Analyze by", ["product", "issue"])
    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(styled_fig(px.bar(top_categories(filtered, dimension, 20), x="complaints", y=dimension, orientation="h", title=f"Top {dimension.title()} Categories"), 520), width="stretch")
    with right:
        grouped = volume_by_month(filtered, dimension)
        top_values = top_categories(filtered, dimension, 5)[dimension].tolist()
        st.plotly_chart(styled_fig(px.line(grouped[grouped[dimension].isin(top_values)], x="month", y="complaints", color=dimension, title=f"Monthly Trend by {dimension.title()}"), 520), width="stretch")
    anomalies = emerging_issue_score(filtered, dimension).head(15)
    st.subheader("Anomaly detection signals")
    st.dataframe(anomalies[[dimension, "complaints", "baseline_mean", "z_score", "emerging_issue_score"]], width="stretch")

elif page == "Company View":
    page_header(
        "Companies",
        "Complaint volume across 1,383 companies",
    )
    bench = company_benchmark(filtered, pd.DataFrame())
    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(styled_fig(px.bar(bench.head(20), x="raw_complaints", y="company", orientation="h", title="Top Companies by Complaint Volume"), 560), width="stretch")
    with right:
        selected_company = st.selectbox("Company detail", bench["company"].head(50).tolist())
        company_df = filtered[filtered["company"] == selected_company]
        metric_card("Selected company complaints", f"{len(company_df):,}", selected_company)
        st.plotly_chart(styled_fig(px.bar(top_categories(company_df, "product", 8), x="complaints", y="product", orientation="h", title="Product Mix"), 390), width="stretch")
    st.dataframe(bench.head(100), width="stretch")

elif page == "Narrative Search":
    page_header(
        "Narratives",
        "Complaint themes and cited narrative search",
    )
    topics = load_artifact_csv(TOPIC_REGISTRY_PATH)
    rag_metrics = load_artifact_json(RAG_EVAL_METRICS_PATH)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Indexed narratives", f"{final_metrics.get('rag_indexed_narratives', 5343):,}", "Searchable complaint text")
    with c2: metric_card("NMF topics", f"{len(topics):,}", "Topic modeling")
    with c3: metric_card("Evidence Recall@5", f"{rag_metrics.get('recall_at_5', 0):.1%}", "Retrieval evaluation")
    if not topics.empty:
        st.subheader("NMF topic summary")
        st.dataframe(topics, width="stretch")

    st.subheader("Narrative search")
    question_bank = [
        "What problems are consumers reporting about incorrect credit report information?",
        "What are consumers saying about debt validation?",
        "What themes show up in credit card payment complaints?",
    ]
    question = st.selectbox("Question", question_bank)
    product_filter = st.selectbox("Product filter", [""] + sorted(df["product"].dropna().unique().tolist()))
    filters = RagFilters(product=product_filter or None)
    if st.button("Search narratives", type="primary"):
        context = build_context(question, df, filters=filters, top_k=5)
        result = generate_answer(context, use_llm=False)
        st.markdown(result["answer"])
        evidence = result.get("evidence", context["evidence"])
        if not evidence.empty:
            visible_cols = ["citation", "date_received", "company", "product", "issue", "clean_narrative"]
            st.dataframe(evidence[visible_cols], width="stretch")

elif page == "Model Performance":
    page_header(
        "Model Performance",
        "Routing model and retrieval evaluation",
    )
    report = load_artifact_json(MODEL_REPORT_PATH)
    rag_metrics = load_artifact_json(RAG_EVAL_METRICS_PATH)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Routing classes", f"{report.get('classes', 10):,}", "Complaint classification")
    with c2: metric_card("Macro F1", f"{report.get('model_macro_f1', 0):.3f}", "TF-IDF Logistic Regression")
    with c3: metric_card("Baseline F1", f"{report.get('baseline_macro_f1', 0):.3f}", "Comparison model")
    with c4: metric_card("Citation validity", f"{rag_metrics.get('citation_validity_rate', 0):.0%}", "Retrieved narratives")
