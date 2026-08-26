from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics.anomaly import detect_spikes, emerging_issue_score
from src.analytics.benchmarking import company_benchmark
from src.analytics.complaints import kpi_summary, top_categories, volume_by_month
from src.config import AI_OPPORTUNITIES_PATH, AI_RISK_REGISTER_PATH, AI_SYSTEM_INVENTORY_PATH, ARTIFACT_DIR, BLS_WAGE_BENCHMARKS_PATH, COMPLAINTS_PARQUET, FRED_ANALYSIS_PATH, FRED_OBSERVATIONS_PATH, GOVERNANCE_CONTROLS_PATH, METADATA_PATH, MODEL_REPORT_PATH, QUALITY_PROFILE_PATH, RAG_EVAL_METRICS_PATH, RAG_EVAL_PATH, RAG_INDEX_PATH, SEC_LINEAGE_PATH, TOPIC_REGISTRY_PATH, VALUE_MONTE_CARLO_PATH, VALUE_SCENARIO_PATH
from src.governance.nist import governance_catalog
from src.governance.inventory import ai_system_inventory, governance_controls, risk_register
from src.nlp.retrieval import retrieve_evidence
from src.rag.assistant import build_context, generate_answer
from src.rag.retrieval import RagFilters
from src.reporting.lineage import lineage_table
from src.storage.duckdb_store import get_store_status
from src.value.model import SCENARIOS, ValueAssumptions, apply_scenario, calculate_value, monte_carlo_summary, monte_carlo_value, scenario_table, sensitivity
from src.value.opportunities import derive_ai_opportunities


st.set_page_config(page_title="CFPB Complaint Lab", page_icon="FS", layout="wide")

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


def money(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}${value/1_000_000:,.1f}M"
    if value >= 1_000:
        return f"{sign}${value/1_000:,.1f}K"
    return f"{sign}${value:,.0f}"


def styled_fig(fig, height: int = 390):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.78)",
        colorway=PLOT_COLORS,
        font={"family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", "color": "#2a2925"},
        title={"font": {"size": 18, "color": "#12120f"}, "x": 0.02, "xanchor": "left"},
        margin={"l": 18, "r": 18, "t": 58, "b": 22},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(103,107,97,.16)", zeroline=False, title_font={"size": 12})
    fig.update_yaxes(showgrid=True, gridcolor="rgba(103,107,97,.16)", zeroline=False, title_font={"size": 12})
    return fig


def page_header(kicker: str, title: str, body: str) -> None:
    st.markdown(f"<div class='eyebrow'>{kicker}</div>", unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"<div class='page-subtitle'>{body}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"<div class='metric-card'><div class='label'>{label}</div><div class='value'>{value}</div><div class='note'>{note}</div></div>",
        unsafe_allow_html=True,
    )


def product_hero(kicker: str, title: str, body: str, rail: list[tuple[str, str]]) -> None:
    rail_html = "".join(f"<div class='rail-card'><strong>{heading}</strong><span>{text}</span></div>" for heading, text in rail)
    st.markdown(
        f"""
        <div class="hero-shell">
          <div class="hero-grid">
            <div class="hero-copy">
              <div class="hero-kicker">{kicker}</div>
              <h1>{title}</h1>
              <p>{body}</p>
            </div>
            <div class="hero-rail">{rail_html}</div>
          </div>
        </div>
        """,
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
    product_hero(
        "Built on real CFPB complaints",
        "A complaint-analysis lab for financial services risk, operations, and automation ideas.",
        "I built this around the public CFPB complaint database because it is messy in the exact way real business data is messy. The app starts with the records, checks the data, finds patterns, and only then talks about where automation might help.",
        [
            ("No made-up companies", "If the CFPB store is missing, the app shows this setup page instead of pretending."),
            ("Messy data is the point", "Narratives are sparse, company names are inconsistent, and category history matters."),
            ("Every number has a trail", "The app keeps source, transformation, assumption, and limitation close to the metric."),
        ],
    )
    st.markdown(
        """
        <div class="command-grid">
          <div class="command-card"><strong>Development refresh</strong><span>Build a fast official CFPB extract for UI review and iteration.</span></div>
          <div class="command-card"><strong>Full refresh</strong><span>Remove the cap when you want the whole public bulk file.</span></div>
          <div class="command-card"><strong>Recent slice</strong><span>Use the API helper when you only need a newer date window.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["Fast start", "Full store", "Recent API slice"])
    with tabs[0]:
        st.code("python -m src.pipeline --max-records 250000", language="bash")
    with tabs[1]:
        st.code("python -m src.pipeline", language="bash")
    with tabs[2]:
        st.code("python -m src.ingestion.cfpb_api --date-min 2026-01-01 --max-pages 25", language="bash")
    st.caption("Large raw and processed files are cached locally and ignored by Git.")
    st.markdown(
        """
        <div class="mosaic">
          <div class="source-tile"><strong>CFPB complaints</strong><span>Dates, products, issues, companies, responses, and public narratives.</span></div>
          <div class="source-tile"><strong>SEC EDGAR</strong><span>Optional scale data for public companies when the match is defensible.</span></div>
          <div class="source-tile"><strong>BLS OEWS</strong><span>Optional wage benchmarks for capacity modeling. Not company payroll data.</span></div>
          <div class="source-tile"><strong>NIST AI RMF</strong><span>A practical checklist for risks, controls, and human review.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


status = get_store_status()
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
        "Portfolio Story",
        "Complaint Trends",
        "Company Benchmarks",
        "Risk Watch",
        "Narrative Topics",
        "Automation Ideas",
        "Value Realization",
        "Scenario Lab",
        "Macro Context",
        "Model Performance",
        "Governance",
        "Evidence Search",
        "Data Notes",
    ],
)

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}

if page == "Overview":
    page_header(
        "Overview",
        "What the complaint data is saying",
        "Start here: volume, coverage, model evidence, and the areas worth investigating before making any automation claim.",
    )
    st.markdown("<div class='warning'>CFPB complaints are useful evidence, but they are not a perfect sample of all consumer experiences. Treat spikes as leads for investigation, not proof of cause.</div>", unsafe_allow_html=True)
    kpi = kpi_summary(filtered)
    if final_metrics:
        signal_strip(
            [
                (f"{final_metrics['model_macro_f1']:.3f}", "Product-routing model macro F1"),
                (f"{final_metrics['topics']:,}", "Topics found in public narratives"),
                (f"{final_metrics['opportunities']:,}", "Automation ideas backed by volume"),
                (f"{final_metrics['rag_groundedness_pass_rate']:.0%}", "Evidence-search citation pass rate"),
                (f"{final_metrics['parquet_size_mb']} MB", "Compressed Parquet store"),
            ]
        )
    cols = st.columns(6)
    with cols[0]: metric_card("Complaints", f"{kpi['complaints']:,}", f"{kpi['date_min']} to {kpi['date_max']}")
    with cols[1]: metric_card("Companies", f"{kpi['companies']:,}", "Observed in selected data")
    with cols[2]: metric_card("Products", f"{kpi['products']:,}", "CFPB taxonomy")
    with cols[3]: metric_card("Issues", f"{kpi['issues']:,}", "Consumer-selected issue")
    with cols[4]: metric_card("Narratives", f"{kpi['narratives']:,}", "Public opt-in narratives")
    with cols[5]: metric_card("Refresh", metadata.get("retrieved_at_utc", "n/a")[:10], "Local store")
    left, right = st.columns([1.4, 1])
    with left:
        monthly = volume_by_month(filtered)
        st.plotly_chart(styled_fig(px.line(monthly, x="month", y="complaints", title="Complaint Volume Over Time"), 420), use_container_width=True)
    with right:
        st.plotly_chart(styled_fig(px.bar(top_categories(filtered, "product", 10), x="complaints", y="product", orientation="h", title="Top Products"), 420), use_container_width=True)
    emerg = emerging_issue_score(filtered, "issue").head(10)
    st.subheader("Worth a closer look")
    st.dataframe(emerg[["issue", "complaints", "baseline_mean", "z_score", "emerging_issue_score"]], use_container_width=True)

elif page == "Portfolio Story":
    page_header(
        "Portfolio story",
        "How this project positions you",
        "This page is built for interviews. It explains the business problem, the analyst decisions, the AI features, and the tradeoffs without sounding rehearsed.",
    )
    if final_metrics:
        signal_strip(
            [
                (f"{final_metrics['record_count']:,}", "Real CFPB records processed"),
                (f"{final_metrics['companies']:,}", "Companies observed"),
                (f"{final_metrics['model_macro_f1']:.3f}", "Routing model macro F1"),
                (f"{final_metrics['opportunities']:,}", "Automation ideas from patterns"),
                (f"{final_metrics['topics']:,}", "Narrative topics"),
            ]
        )
    st.markdown(
        """
        <div class="talk-track">
          <strong>Interview version:</strong>
          I built this because complaint data is a clean way to show business analysis with real operational stakes.
          The data is public, but the problems are the same ones companies deal with internally: messy categories, repeated customer pain,
          incomplete narratives, inconsistent company names, and pressure to automate without losing control.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="story-board">
          <div class="story-card"><small>Business question</small><strong>Where is the operational pain?</strong><p>The app starts with complaint burden by product, issue, company, geography, channel, and response. It avoids raw-count shortcuts by calling out denominator problems.</p></div>
          <div class="story-card"><small>Analytics work</small><strong>What changed, and is it unusual?</strong><p>Monthly trends, rolling baselines, and z-score spikes turn complaint movement into investigation leads instead of vague chart watching.</p></div>
          <div class="story-card"><small>NLP work</small><strong>What are people actually saying?</strong><p>Public narratives are cleaned, grouped into topics, searched by evidence, and used to train a routing classifier with a baseline comparison.</p></div>
          <div class="story-card"><small>AI strategy</small><strong>Where could AI help without being reckless?</strong><p>Automation ideas come from observed patterns. Each one keeps a human role, data requirement, risk, and control attached.</p></div>
          <div class="story-card"><small>Financial thinking</small><strong>What would the business case depend on?</strong><p>The value model uses complaint volume and editable assumptions. It shows when an idea is not financially attractive under baseline assumptions.</p></div>
          <div class="story-card"><small>Governance</small><strong>What would block deployment?</strong><p>The project uses NIST-style risk dimensions to define testing, monitoring, escalation, and review rules before a tool touches a real workflow.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("How to talk about the AI features")
    st.markdown(
        """
        - I did not add AI just to say AI. I used it where complaint operations actually create repeated text and routing problems.
        - The classifier is useful because it is measurable: macro F1 improved from a weak baseline to a much stronger model on public narratives.
        - The evidence search is intentionally boring: complaint IDs, dates, products, issues, and companies come before any written answer.
        - The governance page is part of the product, not a footnote. That is the point.
        """
    )

elif page == "Complaint Trends":
    page_header(
        "Complaint trends",
        "Volume, mix, response, channel, and geography",
        "This is the basic analyst view: pick a dimension, look at the top categories, then check whether the trend actually moved over time.",
    )
    col = st.selectbox("Dimension", ["product", "issue", "company", "state", "submitted_via", "company_response_to_consumer"])
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.plotly_chart(styled_fig(px.bar(top_categories(filtered, col, 20), x="complaints", y=col, orientation="h", title=f"Top {col} categories"), 520), use_container_width=True)
    with c2:
        grouped = volume_by_month(filtered, col)
        top_values = top_categories(filtered, col, 5)[col].tolist()
        st.plotly_chart(styled_fig(px.line(grouped[grouped[col].isin(top_values)], x="month", y="complaints", color=col, title=f"Monthly trend by {col}"), 520), use_container_width=True)
    st.dataframe(filtered.head(1000), use_container_width=True)

elif page == "Company Benchmarks":
    page_header(
        "Company benchmarks",
        "Raw volume first. Normalized views only with caveats.",
        "Large institutions naturally produce more complaints. The app shows raw volume clearly and only normalizes when a public-company scale proxy exists.",
    )
    scale = load_artifact_csv(ARTIFACT_DIR / "sec_company_scale_metrics.csv")
    bench = company_benchmark(filtered, scale)
    sec_lineage = load_artifact_csv(SEC_LINEAGE_PATH)
    if scale.empty:
        st.markdown("<div class='warning'>SEC scale-normalized metrics are not available in this local build. Run entity resolution, then SEC ingestion with a valid SEC_USER_AGENT and network access.</div>", unsafe_allow_html=True)
    else:
        mapped = int(scale.get("cfpb_company_name", pd.Series(dtype=str)).nunique())
        years = int(sec_lineage.get("fiscal_year", pd.Series(dtype=int)).nunique()) if not sec_lineage.empty else 0
        cols = st.columns(3)
        with cols[0]: metric_card("SEC mappings", f"{mapped:,}", "Accepted mapping rows")
        with cols[1]: metric_card("SEC registrants", f"{sec_lineage.get('sec_cik', pd.Series(dtype=str)).nunique() if not sec_lineage.empty else 0:,}", "With observations")
        with cols[2]: metric_card("Fiscal years", f"{years:,}", "Observed in lineage")
    mode = st.radio("View", ["Raw volume", "Normalized view"], horizontal=True)
    if mode == "Raw volume" or "complaints_per_1b_assets" not in bench:
        st.plotly_chart(styled_fig(px.bar(bench.head(20), x="raw_complaints", y="company", orientation="h", title="Raw complaint volume"), 560), use_container_width=True)
    else:
        st.markdown("<div class='warning'>Assets and revenue are transparent scale proxies, not measures of customer count, market share, consumer harm, or company quality.</div>", unsafe_allow_html=True)
        metric = st.selectbox("Normalized metric", ["complaints_per_1b_assets", "complaints_per_1b_revenue"])
        st.plotly_chart(styled_fig(px.bar(bench.dropna(subset=[metric]).head(20), x=metric, y="company", orientation="h", title=metric.replace("_", " ").title()), 560), use_container_width=True)
    st.dataframe(bench, use_container_width=True)
    if not sec_lineage.empty:
        with st.expander("SEC lineage"):
            st.dataframe(sec_lineage, use_container_width=True)

elif page == "Risk Watch":
    page_header(
        "Risk watch",
        "Spikes, baselines, and investigation leads",
        "A z-score is not a smoking gun. It is a way to notice when a product, issue, company, or state deserves a human look.",
    )
    group_col = st.selectbox("Monitor dimension", ["issue", "product", "company", "state"])
    spikes = detect_spikes(filtered, group_col=group_col)
    latest = emerging_issue_score(filtered, group_col=group_col).head(25)
    st.plotly_chart(styled_fig(px.scatter(latest, x="complaints", y="z_score", size="complaints", color=group_col, title="Latest-month spike score"), 520), use_container_width=True)
    st.dataframe(latest, use_container_width=True)
    st.caption("Spike flag uses a z-score >= 2.0 versus the trailing 12-month baseline where enough history exists.")

elif page == "Narrative Topics":
    page_header(
        "Narrative topics",
        "What people are actually writing",
        "The public narratives are sparse, but they are rich. This page groups repeated language and lets you pull examples instead of staring at a word cloud.",
    )
    topics = load_artifact_csv(TOPIC_REGISTRY_PATH)
    if topics.empty:
        st.info("Run `python -m src.pipeline` with NLP enabled to generate the topic registry.")
    else:
        st.dataframe(topics, use_container_width=True)
    query = st.text_input("Retrieve complaint evidence", placeholder="payment issue, account restriction, credit reporting dispute")
    if query:
        st.dataframe(retrieve_evidence(filtered, query), use_container_width=True)

elif page == "Automation Ideas":
    page_header(
        "Automation ideas",
        "Observed problem first. Tool idea second.",
        "This page does not start with a wishlist. It starts with repeated complaint patterns and asks whether a human-in-the-loop tool would reduce friction.",
    )
    opportunities = load_artifact_csv(AI_OPPORTUNITIES_PATH)
    if opportunities.empty:
        opportunities = derive_ai_opportunities(filtered)
    st.dataframe(opportunities, use_container_width=True)
    st.caption("Observed evidence is complaint volume and pattern concentration. The AI intervention is a proposed operating response.")

elif page == "Value Realization":
    page_header(
        "Value model",
        "Value realization from observed complaint workload",
        "Use real complaint volume, then make the assumptions visible: handling time, addressable share, adoption, cost, and success probability.",
    )
    product = st.selectbox("Observed complaint segment", ["All"] + sorted(filtered["product"].dropna().unique().tolist()))
    segment = filtered if product == "All" else filtered[filtered["product"] == product]
    wages = load_artifact_csv(BLS_WAGE_BENCHMARKS_PATH)
    if wages.empty:
        st.markdown("<div class='warning'>No verified BLS wage artifact is present in this local build. The wage below is a user assumption until `python -m src.ingestion.bls_oews` runs successfully.</div>", unsafe_allow_html=True)
        wage = st.number_input("Hourly wage assumption", min_value=1.0, value=24.0, step=1.0)
    else:
        labels = (wages["occupation_code"].astype(str) + " - " + wages["occupation"].astype(str)).tolist()
        selected = st.selectbox("BLS external wage benchmark", labels)
        wage = float(wages.loc[labels.index(selected), "mean_hourly_wage"])
        st.caption("BLS values are external labor-cost benchmarks, not company payroll data.")
    assumptions = ValueAssumptions(observed_complaints=len(segment), hourly_wage=wage)
    output = calculate_value(assumptions)
    cols = st.columns(4)
    with cols[0]: metric_card("Observed complaints", f"{len(segment):,}", "Observed input")
    with cols[1]: metric_card("Hours released", f"{output['expected_hours_released']:,.0f}", "Model output")
    with cols[2]: metric_card("Annual capacity value", money(output["estimated_annual_capacity_value"]), "Model output")
    with cols[3]: metric_card("3-year net value", money(output["three_year_net_value"]), "Under selected assumptions")
    st.json(output)

elif page == "Scenario Lab":
    page_header(
        "Scenario lab",
        "What changes the business case?",
        "Move the assumptions and watch the result change. This is where weak business cases usually reveal themselves.",
    )
    base = ValueAssumptions(
        observed_complaints=st.number_input("Observed complaints", min_value=1, value=int(len(filtered))),
        hourly_wage=st.number_input("Hourly wage benchmark", min_value=1.0, value=24.0),
        average_handling_minutes=st.slider("Average handling minutes", 1.0, 90.0, 22.0),
        ai_addressable_share=st.slider("AI-addressable share", 0.0, 1.0, 0.35),
        time_reduction=st.slider("Time reduction from AI", 0.0, 1.0, 0.25),
        adoption_rate=st.slider("Adoption rate", 0.0, 1.0, 0.70),
        implementation_cost=st.number_input("Implementation cost", min_value=0.0, value=1_250_000.0, step=50_000.0),
        annual_operating_cost=st.number_input("Annual operating cost", min_value=0.0, value=350_000.0, step=25_000.0),
        success_probability=st.slider("Success probability", 0.0, 1.0, 0.75),
    )
    scenario_df = scenario_table(base)
    samples = monte_carlo_value(base, simulations=3000, seed=42)
    mc = monte_carlo_summary(samples)
    tabs = st.tabs(["Scenarios", "Monte Carlo", "Sensitivity"])
    with tabs[0]:
        st.plotly_chart(styled_fig(px.bar(scenario_df, x="scenario", y="three_year_net_value", title="3-year net value by scenario"), 430), use_container_width=True)
        st.dataframe(scenario_df, use_container_width=True)
    with tabs[1]:
        cols = st.columns(4)
        with cols[0]: metric_card("Median NPV", money(mc["median_npv"]), "P50")
        with cols[1]: metric_card("P10 NPV", money(mc["p10_npv"]), "Downside")
        with cols[2]: metric_card("P90 NPV", money(mc["p90_npv"]), "Upside")
        with cols[3]: metric_card("NPV > 0", f"{mc['probability_npv_positive']:.0%}", "Simulation share")
        st.plotly_chart(styled_fig(px.histogram(samples, x="npv", nbins=50, title="Monte Carlo NPV distribution"), 430), use_container_width=True)
        st.caption("Distributions are documented in docs/SCENARIO_METHODOLOGY.md. Modelled value is not realized value.")
    with tabs[2]:
        sens = pd.DataFrame(sensitivity(base))
        st.plotly_chart(styled_fig(px.bar(sens, x="range", y="assumption", orientation="h", title="Tornado sensitivity: 3-year net-value movement"), 470), use_container_width=True)
        st.dataframe(sens, use_container_width=True)

elif page == "Macro Context":
    page_header(
        "Macro context",
        "Complaint trends beside external macro signals",
        "FRED is optional here. It only belongs in the analysis if it adds context without pretending to prove causation.",
    )
    fred = load_artifact_csv(FRED_OBSERVATIONS_PATH)
    fred_analysis = load_artifact_csv(FRED_ANALYSIS_PATH)
    if fred.empty:
        st.info("No FRED observations are present. Configure FRED_API_KEY and run `python -m src.ingestion.fred` if macro overlays are useful.")
    else:
        st.dataframe(fred.tail(1000), use_container_width=True)
    if not fred_analysis.empty:
        st.markdown("<div class='warning'>Correlation does not imply causation. These are exploratory context checks only.</div>", unsafe_allow_html=True)
        st.dataframe(fred_analysis, use_container_width=True)

elif page == "Model Performance":
    page_header(
        "Model check",
        "Narrative routing model performance",
        "A simple baseline comes first. The model has to beat that before it earns a place in an operations workflow.",
    )
    if not MODEL_REPORT_PATH.exists():
        st.info("Run the NLP pipeline to train and evaluate the narrative classifier.")
    else:
        report = json.loads(MODEL_REPORT_PATH.read_text(encoding="utf-8"))
        if report.get("status") == "trained":
            cols = st.columns(3)
            with cols[0]: metric_card("Records", f"{report['records']:,}", "Narratives used")
            with cols[1]: metric_card("Baseline macro F1", f"{report['baseline_macro_f1']:.3f}", "Most-frequent classifier")
            with cols[2]: metric_card("Model macro F1", f"{report['model_macro_f1']:.3f}", "TF-IDF + logistic regression")
        st.json(report)
    if RAG_EVAL_METRICS_PATH.exists():
        st.subheader("RAG evaluation")
        rag_metrics = json.loads(RAG_EVAL_METRICS_PATH.read_text(encoding="utf-8"))
        cols = st.columns(4)
        with cols[0]: metric_card("Eval questions", f"{rag_metrics['question_count']:,}", "Grounded analyst questions")
        with cols[1]: metric_card("Recall@5", f"{rag_metrics['recall_at_5']:.0%}", "Expected evidence in top 5")
        with cols[2]: metric_card("Recall@10", f"{rag_metrics['recall_at_10']:.0%}", "Expected evidence in top 10")
        with cols[3]: metric_card("Citation validity", f"{rag_metrics['citation_validity_rate']:.0%}", "Complaint IDs present")
        st.json(rag_metrics)
        rag_detail_path = RAG_EVAL_METRICS_PATH.with_suffix(".csv")
        if rag_detail_path.exists():
            st.dataframe(pd.read_csv(rag_detail_path), use_container_width=True)
    elif RAG_EVAL_PATH.exists():
        st.subheader("Legacy retrieval evaluation")
        rag = pd.read_csv(RAG_EVAL_PATH)
        cols = st.columns(3)
        with cols[0]: metric_card("Eval questions", f"{len(rag):,}", "Seed analyst questions")
        with cols[1]: metric_card("Citation check", f"{rag['groundedness_gate'].mean():.0%}", "Has complaint IDs")
        with cols[2]: metric_card("Avg term recall", f"{rag['term_recall'].mean():.0%}", "Expected evidence terms")
        st.dataframe(rag, use_container_width=True)

elif page == "Governance":
    page_header(
        "Governance",
        "Controls before deployment",
        "Routing, summarization, and retrieval tools can help. They can also mislead people. This page makes the risks and review rules explicit.",
    )
    opportunities = load_artifact_csv(AI_OPPORTUNITIES_PATH)
    if opportunities.empty:
        opportunities = derive_ai_opportunities(filtered)
    catalog = governance_catalog(opportunities)
    inv = load_artifact_csv(AI_SYSTEM_INVENTORY_PATH)
    risks = load_artifact_csv(AI_RISK_REGISTER_PATH)
    controls = load_artifact_csv(GOVERNANCE_CONTROLS_PATH)
    if inv.empty:
        inv = ai_system_inventory()
    if risks.empty:
        risks = risk_register()
    if controls.empty:
        controls = governance_controls()
    tabs = st.tabs(["System inventory", "Risk register", "Controls", "Opportunity controls"])
    with tabs[0]:
        st.dataframe(inv, use_container_width=True)
    with tabs[1]:
        st.dataframe(risks.sort_values("severity", ascending=False), use_container_width=True)
    with tabs[2]:
        st.dataframe(controls, use_container_width=True)
    with tabs[3]:
        st.dataframe(catalog, use_container_width=True)
    st.caption("NIST AI RMF concepts inform the governance dimensions. This product does not claim compliance certification.")

elif page == "Evidence Search":
    page_header(
        "Grounded RAG",
        "Ask a question. Inspect the answer trail.",
        "The answer is built from deterministic analytics plus retrieved CFPB complaint narratives. If the evidence is thin, it should say that.",
    )
    st.markdown("<div class='warning'>LLMs do not calculate metrics here. Counts, trends, and issue rankings come from local code; generated text must cite complaint evidence or analytics outputs.</div>", unsafe_allow_html=True)
    if not RAG_INDEX_PATH.exists():
        st.error("The RAG index is missing. Run `python scripts/build_rag_index.py` from the project folder.")
        st.stop()

    question_bank = [
        "What problems are consumers reporting about incorrect credit report information?",
        "What are consumers saying about debt validation?",
        "What mortgage servicing issues appear in narratives?",
        "What themes show up in credit card payment complaints?",
        "Which operational problems appear repetitive enough for routing support?",
    ]
    selected_question = st.selectbox("Question starter", ["Custom question"] + question_bank)
    default_question = "" if selected_question == "Custom question" else selected_question
    question = st.text_area("Analyst question", value=default_question, height=95, placeholder="Ask about a product, company, issue, state, or time window.")

    c1, c2, c3 = st.columns(3)
    with c1:
        product_filter = st.selectbox("Product filter", [""] + sorted(df["product"].dropna().unique().tolist()))
        state_filter = st.selectbox("State filter", [""] + sorted(df["state"].dropna().unique().tolist()))
    with c2:
        top_companies = df["company"].value_counts().head(250).index.tolist()
        company_filter = st.selectbox("Company filter", [""] + sorted(top_companies))
        issue_filter = st.selectbox("Issue filter", [""] + sorted(df["issue"].dropna().unique().tolist()))
    with c3:
        min_date, max_date = pd.to_datetime(df["date_received"]).min().date(), pd.to_datetime(df["date_received"]).max().date()
        rag_dates = st.date_input("Evidence date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        top_k = st.slider("Evidence records", min_value=3, max_value=12, value=8)
        use_llm = st.checkbox("Use configured LLM", value=False, help="Uses OPENAI_API_KEY or ANTHROPIC_API_KEY only when LLM_PROVIDER is set. Otherwise the app uses the deterministic grounded fallback.")

    if question:
        date_start = date_end = None
        if isinstance(rag_dates, tuple) and len(rag_dates) == 2:
            date_start, date_end = str(rag_dates[0]), str(rag_dates[1])
        filters = RagFilters(
            company=company_filter or None,
            product=product_filter or None,
            issue=issue_filter or None,
            state=state_filter or None,
            date_start=date_start,
            date_end=date_end,
        )
        try:
            context = build_context(question, df, filters=filters, top_k=top_k)
            result = generate_answer(context, use_llm=use_llm)
        except Exception as exc:
            st.error(f"RAG failed: {exc}")
            st.stop()

        st.subheader("Grounded answer")
        st.markdown(result["answer"])
        st.caption(f"Answer mode: {result['mode']}")

        evidence = result["evidence"]
        st.subheader("Retrieved complaint evidence")
        if evidence.empty:
            st.info("No matching narratives were retrieved for these filters.")
        else:
            visible_cols = ["citation", "date_received", "company", "product", "issue", "state", "hybrid_score", "clean_narrative"]
            st.dataframe(evidence[visible_cols], use_container_width=True)

        st.subheader("Deterministic analytics used")
        st.json(result["analytics"])

        with st.expander("Trace: filters, retrieval, and prompt context"):
            st.json(result["trace"])
            st.text_area("Prompt context sent to LLM or fallback", value=result["prompt_context"], height=260)

elif page == "Data Notes":
    page_header(
        "Data notes",
        "Where the numbers came from",
        "Source, field, transformation, calculation, refresh, and limitation details. Boring on purpose. Useful in interviews.",
    )
    st.subheader("Source metadata")
    st.json(metadata)
    if final_metrics:
        st.subheader("Measured release evidence")
        st.json(final_metrics)
    st.subheader("Lineage")
    st.dataframe(lineage_table(), use_container_width=True)
    if QUALITY_PROFILE_PATH.exists():
        st.subheader("Data quality profile")
        st.dataframe(pd.read_csv(QUALITY_PROFILE_PATH), use_container_width=True)
