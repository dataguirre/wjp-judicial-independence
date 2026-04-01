"""Pre-compute and cache all BERTopic artefacts needed by the Streamlit app.

Saves topics_per_class DataFrames, barchart figures, TPC figures, and color
maps so the dashboard never needs to load a BERTopic model at runtime.

Run once (or after re-running the pipeline) before launching the app:

    uv run python scripts/precompute_topics_per_class.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from bertopic import BERTopic
from transformers import logging as hf_logging

from wjp_judicial_independence.config import PATH_DATA_INTERIM

hf_logging.set_verbosity_error()

COUNTRIES = ["HUNGARY", "ITALY", "POLAND"]
STRATEGIES = ["llm", "embeddings", "llm-api"]

MODULE3_CACHE = PATH_DATA_INTERIM / "module3"
MODULE3_CACHE.mkdir(parents=True, exist_ok=True)

# Fixed palette — must match app.py
TOPIC_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _tpc_path(key: str) -> Path:
    return MODULE3_CACHE / f"tpc_{key}.parquet"


def _fig_path(key: str) -> Path:
    return MODULE3_CACHE / f"fig_{key}.json"


def _color_map_path(key: str) -> Path:
    return MODULE3_CACHE / f"color_map_{key}.json"


def _model_color_map(model: BERTopic) -> dict[int, str]:
    topic_ids = sorted(t for t in model.get_topic_info()["Topic"].tolist() if t != -1)
    return {t: TOPIC_PALETTE[i % len(TOPIC_PALETTE)] for i, t in enumerate(topic_ids)}


def _apply_colors_barchart(fig: go.Figure, color_map: dict[int, str]) -> go.Figure:
    for ann, trace in zip(fig.layout.annotations, fig.data):
        try:
            topic_id = int(ann.text.split()[-1])
            color = color_map.get(topic_id)
            if color:
                trace.marker.color = color
        except (ValueError, IndexError):
            pass
    return fig


def _apply_colors_tpc(fig: go.Figure, color_map: dict[int, str]) -> go.Figure:
    for trace in fig.data:
        try:
            topic_id = int(trace.name.split("_")[0])
            color = color_map.get(topic_id)
            if color:
                trace.marker.color = color
        except (ValueError, AttributeError, IndexError):
            pass
    return fig


def _save_tpc(tpc: pd.DataFrame, key: str) -> None:
    out = _tpc_path(key)
    tpc.to_parquet(out)
    print(f"  Saved TPC → {out.name}")


def _save_fig(fig: go.Figure, key: str) -> None:
    out = _fig_path(key)
    out.write_text(pio.to_json(fig))
    print(f"  Saved fig → {out.name}")


def _save_color_map(color_map: dict[int, str], key: str) -> None:
    out = _color_map_path(key)
    out.write_text(json.dumps({str(k): v for k, v in color_map.items()}))
    print(f"  Saved color map → {out.name}")


def _exists(path: Path) -> bool:
    if path.exists():
        print(f"  Skipping {path.name} (already exists)")
        return True
    return False


# ── Main loop ─────────────────────────────────────────────────────────────────

for strategy in STRATEGIES:
    print(f"\n{'=' * 40}\nStrategy: {strategy}\n{'=' * 40}")

    df = pl.read_parquet(
        PATH_DATA_INTERIM
        / f"module2/sentiment/df_m1_{strategy}_strategy_judicial_independence.parquet"
    )

    # ── General model ────────────────────────────────────────────────────────
    print("Loading general model...")
    general_model = BERTopic.load(
        PATH_DATA_INTERIM / f"module2/topic_modelling/bertopic_general_m1_{strategy}"
    )
    color_map_general = _model_color_map(general_model)

    # Color map
    if not _exists(_color_map_path(f"general_{strategy}")):
        _save_color_map(color_map_general, f"general_{strategy}")

    # Barchart
    if not _exists(_fig_path(f"bar_general_{strategy}")):
        fig = general_model.visualize_barchart(
            width=600, height=400, title="Top Topics"
        )
        _apply_colors_barchart(fig, color_map_general)
        _save_fig(fig, f"bar_general_{strategy}")

    # TPC — country
    tpc_key = f"general_{strategy}_country"
    if not _exists(_tpc_path(tpc_key)):
        print(f"  Computing TPC {tpc_key}...")
        tpc = general_model.topics_per_class(
            df["event"].to_list(), classes=df["country"].to_list()
        )
        _save_tpc(tpc, tpc_key)
    else:
        tpc = pd.read_parquet(_tpc_path(tpc_key))

    if not _exists(_fig_path(f"tpc_general_{strategy}_country")):
        fig = general_model.visualize_topics_per_class(
            tpc,
            title="General Topics per Country",
            width=500,
            height=400,
            normalize_frequency=False,
        )
        fig.update_traces(visible=True)
        _apply_colors_tpc(fig, color_map_general)
        fig.update_layout(
            showlegend=False,
            height=fig.layout.height,
        )
        _save_fig(fig, f"tpc_general_{strategy}_country")

    # ── Per-country models ───────────────────────────────────────────────────
    for country in COUNTRIES:
        print(f"\nLoading {country} model...")
        model = BERTopic.load(
            PATH_DATA_INTERIM
            / f"module2/topic_modelling/bertopic_{country}_m1_{strategy}"
        )
        df_c = df.filter(pl.col("country") == country)
        color_map_c = _model_color_map(model)

        # Color map
        if not _exists(_color_map_path(f"{country}_{strategy}")):
            _save_color_map(color_map_c, f"{country}_{strategy}")

        # Barchart
        if not _exists(_fig_path(f"bar_{country}_{strategy}")):
            fig = model.visualize_barchart(width=400, height=350, title="Topics")
            _apply_colors_barchart(fig, color_map_c)
            _save_fig(fig, f"bar_{country}_{strategy}")

        # TPC — sentiment
        tpc_key_sent = f"{country}_{strategy}_sentiment"
        if not _exists(_tpc_path(tpc_key_sent)):
            print(f"  Computing TPC {tpc_key_sent}...")
            tpc_sent = model.topics_per_class(
                df_c["event"].to_list(),
                classes=df_c["judicial_independence_sentiment"].to_list(),
            )
            _save_tpc(tpc_sent, tpc_key_sent)
        else:
            tpc_sent = pd.read_parquet(_tpc_path(tpc_key_sent))

        if not _exists(_fig_path(f"tpc_{country}_{strategy}_sentiment")):
            tpc_filtered = tpc_sent[tpc_sent["Class"] != "neutral"]
            fig = model.visualize_topics_per_class(
                tpc_filtered,
                title="Topics per Judicial Independence Sentiment",
                width=1200,
                height=400,
                normalize_frequency=False,
            )
            fig.update_traces(visible=True)
            _apply_colors_tpc(fig, color_map_c)
            fig.update_layout(showlegend=False)
            _save_fig(fig, f"tpc_{country}_{strategy}_sentiment")

        # TPC — pillar
        tpc_key_pillar = f"{country}_{strategy}_pillar"
        if not _exists(_tpc_path(tpc_key_pillar)):
            print(f"  Computing TPC {tpc_key_pillar}...")
            tpc_pillar = model.topics_per_class(
                df_c["event"].to_list(),
                classes=df_c["pillar"].to_list(),
            )
            _save_tpc(tpc_pillar, tpc_key_pillar)
        else:
            tpc_pillar = pd.read_parquet(_tpc_path(tpc_key_pillar))

        if not _exists(_fig_path(f"tpc_{country}_{strategy}_pillar")):
            fig = model.visualize_topics_per_class(
                tpc_pillar,
                title="Topics per WJP Pillar",
                width=800,
                height=700,
                normalize_frequency=False,
            )
            fig.update_traces(visible=True)
            _apply_colors_tpc(fig, color_map_c)
            fig.update_layout(showlegend=False)
            _save_fig(fig, f"tpc_{country}_{strategy}_pillar")

print("\nDone. All artefacts cached in:", MODULE3_CACHE)
