"""Pre-compute and cache topics_per_class DataFrames for all strategies and countries.

Run once before launching the Streamlit app so it never needs to recompute
these expensive BERTopic calls at runtime:

    uv run python scripts/precompute_topics_per_class.py
"""

from pathlib import Path
from bertopic import BERTopic
import polars as pl
from transformers import logging as hf_logging

from wjp_judicial_independence.config import PATH_DATA_INTERIM

hf_logging.set_verbosity_error()

COUNTRIES = ["HUNGARY", "ITALY", "POLAND"]
STRATEGIES = ["llm", "embeddings", "llm-api"]

MODULE3_CACHE = PATH_DATA_INTERIM / "module3"
MODULE3_CACHE.mkdir(parents=True, exist_ok=True)


for strategy in STRATEGIES:
    print(f"\n{'='*40}\nStrategy: {strategy}\n{'='*40}")

    df = pl.read_parquet(
        PATH_DATA_INTERIM / f"module2/sentiment/df_m1_{strategy}_strategy_judicial_independence.parquet"
    )

    # ── General model ────────────────────────────────────────────────────────
    print("Loading general model...")
    general_model = BERTopic.load(
        PATH_DATA_INTERIM / f"module2/topic_modelling/bertopic_general_m1_{strategy}"
    )

    key = f"general_{strategy}_country"
    out = MODULE3_CACHE / f"tpc_{key}.parquet"
    if not out.exists():
        print(f"  Computing {key}...")
        tpc = general_model.topics_per_class(df["event"].to_list(), classes=df["country"].to_list())
        tpc.to_parquet(out)
        print(f"  Saved → {out}")
    else:
        print(f"  Skipping {key} (already exists)")

    # ── Country models ───────────────────────────────────────────────────────
    for country in COUNTRIES:
        print(f"Loading {country} model...")
        model = BERTopic.load(
            PATH_DATA_INTERIM / f"module2/topic_modelling/bertopic_{country}_m1_{strategy}"
        )
        df_c = df.filter(pl.col("country") == country)

        for class_col, suffix in [
            ("judicial_independence_sentiment", "sentiment"),
            ("pillar", "pillar"),
        ]:
            key = f"{country}_{strategy}_{suffix}"
            out = MODULE3_CACHE / f"tpc_{key}.parquet"
            if not out.exists():
                print(f"  Computing {key}...")
                tpc = model.topics_per_class(
                    df_c["event"].to_list(),
                    classes=df_c[class_col].to_list(),
                )
                tpc.to_parquet(out)
                print(f"  Saved → {out}")
            else:
                print(f"  Skipping {key} (already exists)")

print("\nDone. All topics_per_class files are cached in:", MODULE3_CACHE)
