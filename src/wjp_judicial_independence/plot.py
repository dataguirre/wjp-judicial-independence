import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import polars.selectors as cs
from plotly.subplots import make_subplots
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIRECTION_COLORS = {
    "strengthening": "#1D9E75",
    "neutral": "#888780",
    "threat": "#D85A30",
}

_IMPACT_COLORS = {
    "Very Positive": "#E1F5EE",
    "Positive": "#9FE1CB",
    "Neutral": "#F1EFE8",
    "Negative": "#FAECE7",
    "Very Negative": "#F5C4B3",
}

_IMPACT_ORDER = ["Very Negative", "Negative", "Very Positive", "Positive", "Neutral"]

_TABLE_HEADER_COLOR = "#2C2C2A"
_TABLE_FILL_COLORS = {
    "strengthening": "#E1F5EE",
    "threat": "#FAECE7",
    "neutral": "#F1EFE8",
}


# ---------------------------------------------------------------------------
# Module I
# ---------------------------------------------------------------------------


def plot_events_by_pillar(df: pl.DataFrame) -> None:
    """Plot event counts by pillar, split by judicial independence relevance.

    Displays a grouped bar chart with one pair of bars per pillar: one for
    events classified as relevant to judicial independence and one for the rest.
    Value labels are rendered above each bar.

    Args:
        df: DataFrame with at least ``pillar`` and ``is_judicial_independence``
            columns, as returned by
            :func:`~wjp_judicial_independence.classifier.classify_events`.
    """
    counts = (
        df.group_by("pillar", "is_judicial_independence")
        .len()
        .pivot(on="is_judicial_independence", index="pillar", values="len")
        .rename({"true": "Judicial Independence", "false": "Other events"})
        .sort("pillar")
        .fill_null(0)
        .to_pandas()
        .set_index("pillar")
    )

    x = np.arange(len(counts))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_other = ax.bar(
        x - width / 2,
        counts["Other events"],
        width,
        label="Other events",
        color="#B4B2A9",
        alpha=0.9,
    )
    bars_ji = ax.bar(
        x + width / 2,
        counts["Judicial Independence"],
        width,
        label="Judicial Independence",
        color="#378ADD",
        alpha=0.9,
    )

    for bar in (*bars_other, *bars_ji):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Wrap long pillar names so they don't overlap
    labels = [label.replace(" ", "\n") for label in counts.index]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Number of events")
    ax.set_xlabel("WJP Pillar")
    ax.set_title("Judicial Independence events by WJP pillar")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()


def plot_comparison(
    overall: pl.DataFrame,
    by_pillar: pl.DataFrame,
    by_country: pl.DataFrame,
    strategies: list[str],
) -> None:
    """Render three comparison plots: overall, per-pillar, and per-country JI rates."""
    colors = ["#378ADD", "#E07B3F", "#4CAF50"][: len(strategies)]
    x_pad = np.arange(len(strategies))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Judicial Independence — Strategy Comparison", fontsize=13)

    # --- Overall ---
    ax = axes[0]
    bars = ax.bar(
        x_pad, overall["ji_rate"].to_list(), color=colors, alpha=0.9, width=0.5
    )
    for bar, count in zip(bars, overall["ji_count"].to_list()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x_pad)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.set_ylabel("JI rate")
    ax.set_title("Overall JI rate")
    ax.set_ylim(0, overall["ji_rate"].max() * 1.2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # --- Per pillar ---
    ax = axes[1]
    pillars = sorted(by_pillar["pillar"].unique().to_list())
    x = np.arange(len(pillars))
    width = 0.8 / len(strategies)
    offsets = np.linspace(
        -(len(strategies) - 1) / 2, (len(strategies) - 1) / 2, len(strategies)
    )

    for offset, strategy, color in zip(offsets, strategies, colors):
        rates = (
            by_pillar.filter(pl.col("strategy") == strategy)
            .sort("pillar")["ji_rate"]
            .to_list()
        )
        ax.bar(x + offset * width, rates, width, label=strategy, color=color, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace(" ", "\n") for p in pillars], fontsize=8)
    ax.set_ylabel("JI rate")
    ax.set_title("JI rate by pillar")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # --- Per country ---
    ax = axes[2]
    countries = sorted(by_country["country"].unique().to_list())
    x = np.arange(len(countries))

    for offset, strategy, color in zip(offsets, strategies, colors):
        rates = (
            by_country.filter(pl.col("strategy") == strategy)
            .sort("country")["ji_rate"]
            .to_list()
        )
        ax.bar(x + offset * width, rates, width, label=strategy, color=color, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(countries, fontsize=9)
    ax.set_ylabel("JI rate")
    ax.set_title("JI rate by country")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Module II
# ---------------------------------------------------------------------------


def plot_ji_sentiment(
    df: pl.DataFrame,
    output_html: str | None = None,
) -> go.Figure:
    """Plot judicial independence sentiment by country.

    Renders two side-by-side horizontal bar charts:

    - Left: stacked proportional bar showing the share of strengthening /
      neutral / threat events per country.
    - Right: JI Index — ``(strengthening − threat) / total`` per country.

    Args:
        df: DataFrame with at least ``is_judicial_independence``,
            ``country``, and ``judicial_independence_sentiment`` columns,
            as returned by
            :func:`~wjp_judicial_independence.sentiment.classify_sentiment`.
        output_html: Optional file path. When provided the figure is also
            saved as a standalone HTML file.

    Returns:
        The :class:`plotly.graph_objects.Figure` object.
    """
    df_ji = df.filter(pl.col("is_judicial_independence")).to_pandas()

    counts = (
        df_ji.groupby(["country", "judicial_independence_sentiment"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ["strengthening", "neutral", "threat"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["strengthening", "neutral", "threat"]]
    counts_pct = counts.div(counts.sum(axis=1), axis=0) * 100

    country_order = counts_pct.index.tolist()

    scorecard = {}
    for country in country_order:
        df_c = df_ji[df_ji["country"] == country]
        total = len(df_c)
        s = (df_c["judicial_independence_sentiment"] == "strengthening").sum()
        t = (df_c["judicial_independence_sentiment"] == "threat").sum()
        scorecard[country] = round((s - t) / total, 3) if total > 0 else 0

    scorecard_s = pd.Series(scorecard).reindex(country_order)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Judicial Independence Sentiment by Country (%)",
            "Judicial Independence Index by Country",
        ),
        horizontal_spacing=0.08,
    )

    for direction in ["strengthening", "neutral", "threat"]:
        fig.add_trace(
            go.Bar(
                name=direction.capitalize(),
                y=counts_pct.index.tolist(),
                x=counts_pct[direction].tolist(),
                orientation="h",
                marker_color=_DIRECTION_COLORS[direction],
                text=[f"{v:.0f}%" if v > 8 else "" for v in counts_pct[direction]],
                textposition="inside",
                insidetextanchor="middle",
                legendgroup=direction,
            ),
            row=1,
            col=1,
        )

    fig.add_vline(x=50, line_dash="dash", line_color="gray", line_width=1, row=1, col=1)
    fig.update_xaxes(range=[0, 100], title_text="Percentage (%)", row=1, col=1)

    bar_colors = [
        _DIRECTION_COLORS["threat"] if v < 0 else _DIRECTION_COLORS["strengthening"]
        for v in scorecard_s.values
    ]

    fig.add_trace(
        go.Bar(
            name="JI Index",
            y=scorecard_s.index.tolist(),
            x=scorecard_s.values.tolist(),
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:+.2f}" for v in scorecard_s.values],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_vline(x=0, line_color="black", line_width=1, row=1, col=2)
    fig.update_xaxes(
        range=[-1, 1], title_text="(strengthening − threat) / total", row=1, col=2
    )

    fig.update_yaxes(categoryorder="array", categoryarray=country_order, row=1, col=1)
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=country_order,
        row=1,
        col=2,
        showticklabels=False,
    )

    fig.update_layout(
        barmode="stack",
        height=400,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="#E5E5E5")

    fig.show()
    if output_html:
        fig.write_html(output_html)

    return fig


def _build_wordcloud(
    topics_per_class: pd.DataFrame, sentiment: str, color: str
) -> WordCloud:
    word_freq: dict[str, int] = {}
    for _, row in topics_per_class[topics_per_class["Class"] == sentiment].iterrows():
        for word in str(row["Words"]).split(","):
            word = word.strip()
            if word:
                word_freq[word] = word_freq.get(word, 0) + int(row["Frequency"])

    return WordCloud(
        width=700,
        height=300,
        background_color="white",
        color_func=lambda *args, **kwargs: color,
        max_words=80,
        collocations=False,
    ).generate_from_frequencies(word_freq)


def plot_sentiment_wordclouds(
    topics_per_class_per_country: list[tuple[pd.DataFrame, str]],
) -> None:
    """Plot threat and strengthening word clouds for multiple countries.

    Renders one row per country, with threat on the left and strengthening on
    the right. The country name is shown as a left-side row label.

    Args:
        topics_per_class_per_country: List of ``(topics_per_class, country)``
            tuples. ``topics_per_class`` is the DataFrame returned by
            ``BERTopic.topics_per_class`` with ``Words``, ``Frequency``, and
            ``Class`` columns. Neutral rows are ignored.
    """
    n = len(topics_per_class_per_country)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=(14, 4 * n),
        gridspec_kw={"width_ratios": [0.08, 1, 1]},
    )
    if n == 1:
        axes = [axes]

    for ax_row, (topics_per_class, country) in zip(axes, topics_per_class_per_country):
        df = topics_per_class[topics_per_class["Class"] != "neutral"]

        wc_threat = _build_wordcloud(df, "threat", _DIRECTION_COLORS["threat"])
        wc_strengthening = _build_wordcloud(
            df, "strengthening", _DIRECTION_COLORS["strengthening"]
        )

        ax_row[0].axis("off")
        ax_row[0].text(
            0.5,
            0.5,
            country.capitalize(),
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            rotation=90,
            transform=ax_row[0].transAxes,
        )

        ax_row[1].imshow(wc_threat, interpolation="bilinear")
        ax_row[1].axis("off")
        ax_row[1].set_title(
            "Threat", fontsize=11, color=_DIRECTION_COLORS["threat"], pad=6
        )

        ax_row[2].imshow(wc_strengthening, interpolation="bilinear")
        ax_row[2].axis("off")
        ax_row[2].set_title(
            "Strengthening",
            fontsize=11,
            color=_DIRECTION_COLORS["strengthening"],
            pad=6,
        )

    plt.suptitle("Word Clouds — Judicial Independence Sentiment", fontsize=16, y=1.01)
    plt.tight_layout(h_pad=3)
    plt.show()


# ---------------------------------------------------------------------------
# Module III
# ---------------------------------------------------------------------------


def plot_critical_events(
    df: pl.DataFrame,
    top_n: int = 5,
    output_html: str | None = None,
) -> go.Figure:
    """Plot the top N critical judicial independence events per country as tables.

    Filters out neutral events, ranks by the highest score across all score
    columns, and displays one table per country stacked vertically.

    Args:
        df: DataFrame with at least ``country``, ``pillar``,
            ``judicial_independence_sentiment``, ``event``, and one or more
            ``score_*`` columns.
        top_n: Number of top events to show per country.
        output_html: Optional path to save the figure as a standalone HTML file.

    Returns:
        The :class:`plotly.graph_objects.Figure` object.
    """
    countries = df["country"].unique().sort().to_list()

    top_events = (
        df.with_columns(max_score=pl.max_horizontal(cs.starts_with("score")))
        .filter(pl.col("judicial_independence_sentiment") != "neutral")
        .sort("max_score", descending=True)
        .group_by("country")
        .head(top_n)
    )

    _row_h = 60
    _header_h = 32
    _title_h = 40
    _rows_per_country = [
        len(top_events.filter(pl.col("country") == c)) for c in countries
    ]
    _per_table = [_header_h + n * _row_h + _title_h for n in _rows_per_country]
    _total_h = sum(_per_table) + 80

    fig = make_subplots(
        rows=len(countries),
        cols=1,
        subplot_titles=[f"Top {top_n} Critical Events — {c}" for c in countries],
        vertical_spacing=0.03,
        specs=[[{"type": "table"}] for _ in countries],
        row_heights=_per_table,
    )

    for i, country in enumerate(countries):
        df_c = top_events.filter(pl.col("country") == country).sort(
            "max_score", descending=True
        )

        sentiments = df_c["judicial_independence_sentiment"].to_list()
        cell_colors = [
            [_TABLE_FILL_COLORS[s] for s in sentiments],
            [_TABLE_FILL_COLORS[s] for s in sentiments],
            [_TABLE_FILL_COLORS[s] for s in sentiments],
            [_TABLE_FILL_COLORS[s] for s in sentiments],
        ]

        fig.add_trace(
            go.Table(
                columnwidth=[2, 2, 1, 5],
                header=dict(
                    values=[
                        "<b>Pillar</b>",
                        "<b>Judicial Independence Sentiment</b>",
                        "<b>Score</b>",
                        "<b>Event</b>",
                    ],
                    fill_color=_TABLE_HEADER_COLOR,
                    font=dict(color="white", size=12),
                    align=["left", "left", "center", "left"],
                    height=32,
                ),
                cells=dict(
                    values=[
                        df_c["pillar"].to_list(),
                        df_c["judicial_independence_sentiment"].to_list(),
                        [f"{v:.3f}" for v in df_c["max_score"].to_list()],
                        df_c["event"].to_list(),
                    ],
                    fill_color=cell_colors,
                    font=dict(color=_TABLE_HEADER_COLOR, size=11),
                    align=["left", "left", "center", "left"],
                    height=60,
                ),
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        title_text=f"Critical Judicial Independence Events by Country (Top {top_n})",
        title_font_size=14,
        height=_total_h,
        paper_bgcolor="white",
    )

    fig.show()
    if output_html:
        fig.write_html(output_html)

    return fig


def plot_interesting_events(
    df: pl.DataFrame,
    top_n: int | None = None,
    output_html: str | None = None,
) -> go.Figure:
    """Plot events where impact and judicial independence sentiment diverge.

    "Interesting" events are those where the media framing (impact) contradicts
    the judicial independence sentiment — e.g. a positive news story that is
    actually a threat, or a negative story that strengthens independence.

    Args:
        df: DataFrame with at least ``country``, ``pillar``, ``impact``,
            ``judicial_independence_sentiment``, ``event``, and one or more
            ``score_*`` columns.
        top_n: Number of top events to show per country.
        output_html: Optional path to save the figure as a standalone HTML file.

    Returns:
        The :class:`plotly.graph_objects.Figure` object.
    """
    interesting_news = (
        (pl.col("impact").is_in(["Positive", "Very Positive", "Neutral"]))
        & (pl.col("judicial_independence_sentiment") == "threat")
    ) | (
        (pl.col("impact").is_in(["Negative", "Very Negative", "Neutral"]))
        & (pl.col("judicial_independence_sentiment") == "strengthening")
    )

    df_interesting = (
        df.with_columns(max_score=pl.max_horizontal(cs.starts_with("score")))
        .filter(interesting_news)
        .sort("max_score", descending=True)
    )

    countries = df_interesting["country"].unique().sort().to_list()

    top_interesting = (
        df_interesting.group_by("country").head(top_n)
        if top_n is not None
        else df_interesting
    )

    _row_h = 60
    _header_h = 32
    _title_h = 40
    _rows_per_country = [
        len(top_interesting.filter(pl.col("country") == c)) for c in countries
    ]
    # Content height per table (header + data rows + subplot title)
    _per_table = [_header_h + n * _row_h + _title_h for n in _rows_per_country]
    _total_h = sum(_per_table) + 80

    fig = make_subplots(
        rows=len(countries),
        cols=1,
        subplot_titles=[f"Interesting Events — {c}" for c in countries],
        vertical_spacing=0.03,
        specs=[[{"type": "table"}] for _ in countries],
        row_heights=_per_table,
    )

    for i, country in enumerate(countries):
        df_c = (
            top_interesting.filter(pl.col("country") == country)
            .with_columns(
                pl.col("impact").cast(pl.Enum(_IMPACT_ORDER)).alias("_impact_rank")
            )
            .sort(["_impact_rank", "max_score"], descending=[False, True])
            .drop("_impact_rank")
        )

        sentiments = df_c["judicial_independence_sentiment"].to_list()
        impacts = df_c["impact"].to_list()

        cell_colors = [
            [_IMPACT_COLORS.get(imp, "#F1EFE8") for imp in impacts],
            [_IMPACT_COLORS.get(imp, "#F1EFE8") for imp in impacts],
            [_TABLE_FILL_COLORS.get(s, "#F1EFE8") for s in sentiments],
            [_TABLE_FILL_COLORS.get(s, "#F1EFE8") for s in sentiments],
        ]

        fig.add_trace(
            go.Table(
                columnwidth=[2, 1, 2, 5],
                header=dict(
                    values=[
                        "<b>Pillar</b>",
                        "<b>Impact</b>",
                        "<b>Judicial Independence Sentiment</b>",
                        "<b>Event</b>",
                    ],
                    fill_color=_TABLE_HEADER_COLOR,
                    font=dict(color="white", size=12),
                    align=["left", "left", "left", "left"],
                    height=32,
                ),
                cells=dict(
                    values=[
                        df_c["pillar"].to_list(),
                        df_c["impact"].to_list(),
                        df_c["judicial_independence_sentiment"].to_list(),
                        df_c["event"].to_list(),
                    ],
                    fill_color=cell_colors,
                    font=dict(color=_TABLE_HEADER_COLOR, size=11),
                    align=["left", "left", "left", "left"],
                    height=60,
                ),
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        title_text=f"Interesting Events by Country{f' (Top {top_n})' if top_n else ''}",
        title_font_size=14,
        height=_total_h,
        paper_bgcolor="white",
    )

    fig.show()
    if output_html:
        fig.write_html(output_html)

    return fig
