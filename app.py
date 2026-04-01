"""WJP Judicial Independence — Streamlit Dashboard.

Replicates the analysis from notebooks/module3_visualization_and_analysis.ipynb
(sections 1 & 2) with interactive controls.

All BERTopic artefacts (figures, color maps, TPC DataFrames) are pre-computed
by scripts/precompute_topics_per_class.py — no BERTopic needed at runtime.
"""

# ── Suppress interactive display before any imports that trigger rendering ───
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.basedatatypes as _pbd

_pbd.BaseFigure.show = lambda self, *args, **kwargs: None  # no-op in Streamlit

# ── Standard imports ─────────────────────────────────────────────────────────
import json

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import streamlit as st

from wjp_judicial_independence.config import PATH_DATA_INTERIM
from wjp_judicial_independence.plot import (
    _DIRECTION_COLORS,
    plot_critical_events,
    plot_interesting_events,
    plot_ji_sentiment,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WJP Judicial Independence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


COUNTRIES = ["HUNGARY", "ITALY", "POLAND"]
STRATEGY_LABELS = {
    "llm": "LLM (local)",
    "embeddings": "Embeddings",
    "llm-api": "LLM API",
}

MODULE3_CACHE = PATH_DATA_INTERIM / "module3"


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data(strategy: str) -> pl.DataFrame:
    return pl.read_parquet(
        PATH_DATA_INTERIM
        / f"module2/sentiment/df_m1_{strategy}_strategy_judicial_independence.parquet"
    )


@st.cache_data(show_spinner=False)
def load_figure(key: str) -> go.Figure:
    path = MODULE3_CACHE / f"fig_{key}.json"
    return pio.from_json(path.read_text())


@st.cache_data(show_spinner=False)
def load_color_map(key: str) -> dict[int, str]:
    path = MODULE3_CACHE / f"color_map_{key}.json"
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


@st.cache_data(show_spinner=False)
def load_tpc(cache_key: str) -> pd.DataFrame:
    return pd.read_parquet(MODULE3_CACHE / f"tpc_{cache_key}.parquet")


# ── Word cloud grid ───────────────────────────────────────────────────────────
def _plot_wordclouds_grid(
    tpc_per_country: list[tuple[pd.DataFrame, str]],
) -> plt.Figure:
    """Word clouds in a 2-row × N-country grid.

    Row 0 = Threat, Row 1 = Strengthening. Countries as columns.
    """
    from wordcloud import WordCloud

    _WC_W, _WC_H = 560, 360

    def _make_wc(tpc_df: pd.DataFrame, sentiment: str, color: str):
        word_freq: dict[str, int] = {}
        for _, row in tpc_df[tpc_df["Class"] == sentiment].iterrows():
            for word in str(row["Words"]).split(","):
                word = word.strip()
                if word:
                    word_freq[word] = word_freq.get(word, 0) + int(row["Frequency"])
        if not word_freq:
            return None
        return WordCloud(
            width=_WC_W,
            height=_WC_H,
            background_color="white",
            color_func=lambda *args, **kwargs: color,
            max_words=80,
            collocations=False,
        ).generate_from_frequencies(word_freq)

    sentiments = [
        ("threat", _DIRECTION_COLORS["threat"]),
        ("strengthening", _DIRECTION_COLORS["strengthening"]),
    ]
    n = len(tpc_per_country)

    _cell_w = 1.2
    _cell_h = _cell_w * (_WC_H / _WC_W)
    _fig_h = _cell_h * 2 + 0.25

    fig, axes = plt.subplots(2, n, figsize=(_cell_w * n, _fig_h), dpi=900)

    for col, (tpc, country) in enumerate(tpc_per_country):
        df = tpc[tpc["Class"] != "neutral"]
        axes[0, col].set_title(
            country.capitalize(), fontsize=5, fontweight="bold", pad=3
        )

        for row, (sentiment, color) in enumerate(sentiments):
            wc = _make_wc(df, sentiment, color)
            ax = axes[row, col]
            if wc:
                ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")

    plt.subplots_adjust(
        hspace=0.1, wspace=0.1, top=0.90, bottom=0.02, left=0.05, right=0.98
    )

    for row, (sentiment, color) in enumerate(sentiments):
        pos = axes[row, 0].get_position()
        fig.text(
            0.0001,
            (pos.y0 + pos.y1) / 2,
            sentiment.capitalize(),
            va="center",
            ha="left",
            fontsize=5,
            color=color,
            fontweight="bold",
            rotation=90,
        )

    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/wjp_logo_text.svg")
    st.subheader("Judicial Independence Dashboard")
    st.divider()

    strategy = st.selectbox(
        "Module I Strategy",
        options=["llm", "embeddings", "llm-api"],
        format_func=lambda x: STRATEGY_LABELS[x],
        help="Classification strategy used in Module I to identify JI-relevant events.",
    )

    top_n = st.slider(
        "Top N Critical Events per Country",
        min_value=1,
        max_value=20,
        value=5,
    )

    country = st.selectbox(
        "Country (Section 1.3 Deep Dive)",
        options=COUNTRIES,
        index=2,
        format_func=str.capitalize,
    )


# ── Load main dataframe ───────────────────────────────────────────────────────
df = load_data(strategy)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_topics, tab_events = st.tabs(
    [
        "1. Topic Distribution",
        "2. Critical & Interesting Events",
    ]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — TOPIC DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════
with tab_topics:
    # ── 1.1 General Topics ───────────────────────────────────────────────────
    st.subheader("1.1 General Topics in Judicial Independence")

    col_bar, col_tpc = st.columns([1, 1])

    with col_bar:
        fig_general_bar = load_figure(f"bar_general_{strategy}")
        st.plotly_chart(fig_general_bar, use_container_width=True)

    with col_tpc:
        fig_tpc_country = load_figure(f"tpc_general_{strategy}_country")
        fig_tpc_country.update_layout(height=fig_general_bar.layout.height)
        st.plotly_chart(fig_tpc_country, use_container_width=True)

    st.divider()

    # ── 1.2 JI Sentiment Comparison ─────────────────────────────────────────
    st.subheader("1.2 Judicial Independence Sentiment Comparison by Country")

    fig_ji = plot_ji_sentiment(df)
    st.plotly_chart(fig_ji, use_container_width=True)

    st.markdown("**Sentiment Word Clouds per Country**")
    st.caption(
        "Topics associated with threat vs. strengthening events for each country."
    )

    tpc_per_country = [(load_tpc(f"{c}_{strategy}_sentiment"), c) for c in COUNTRIES]

    fig_wc = _plot_wordclouds_grid(tpc_per_country)
    _, col_wc, _ = st.columns([1, 3, 1])
    with col_wc:
        st.pyplot(fig_wc, use_container_width=True)
    plt.close(fig_wc)

    st.divider()

    # ── 1.3 Country Deep Dive ────────────────────────────────────────────────
    st.subheader(f"1.3 Topics in Judicial Independence — {country.capitalize()}")

    col_c_bar, col_c_pillar = st.columns([1, 1])

    with col_c_bar:
        fig_c_bar = load_figure(f"bar_{country}_{strategy}")
        st.plotly_chart(fig_c_bar, use_container_width=True)

    with col_c_pillar:
        fig_c_pillar = load_figure(f"tpc_{country}_{strategy}_pillar")
        fig_c_pillar.update_layout(height=fig_c_bar.layout.height)
        st.plotly_chart(fig_c_pillar, use_container_width=True)

    fig_c_sent = load_figure(f"tpc_{country}_{strategy}_sentiment")
    st.plotly_chart(fig_c_sent, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CRITICAL & INTERESTING EVENTS
# ════════════════════════════════════════════════════════════════════════════
with tab_events:
    # ── 2.1 Critical Events ──────────────────────────────────────────────────
    st.subheader(f"2.1 Critical Events (Top {top_n} per Country)")
    st.caption(
        "Events with the highest judicial independence score, "
        "excluding neutral sentiment."
    )

    fig_critical = plot_critical_events(df, top_n=top_n)
    st.plotly_chart(fig_critical, use_container_width=True)

    st.divider()

    # ── 2.2 Interesting Events ───────────────────────────────────────────────
    st.subheader("2.2 Interesting Events — Divergent Impact & JI Sentiment")
    st.caption(
        "Events where media framing (impact) contradicts JI sentiment: "
        "positive news that is a threat, or negative news that strengthens independence."
    )

    fig_interesting = plot_interesting_events(df)
    st.plotly_chart(fig_interesting, use_container_width=True)
