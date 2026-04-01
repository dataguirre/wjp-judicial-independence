"""WJP pipeline: Module 1 (classification) + Module 2 (sentiment + topic modeling).

Usage
-----
# Run default strategies (embeddings + llm)
uv run python scripts/pipeline.py

# Choose specific strategies
uv run python scripts/pipeline.py --strategies embeddings llm llm-api \\
    --api-provider openai --api-key sk-... --api-model gpt-4o-mini

# Force re-run even if outputs already exist
uv run python scripts/pipeline.py --force
"""

import argparse
import logging
import os
import sys

import polars as pl
import polars.selectors as cs
from transformers import logging as hf_logging

from wjp_judicial_independence.config import PATH_DATA_INTERIM, PATH_DATA_RAW
from wjp_judicial_independence.classifier import classify_events
from wjp_judicial_independence.preprocessing import load_events
from wjp_judicial_independence.sentiment import classify_sentiment

hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-mpnet-base-v2"
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

REFERENCES = {
    "compliance_with_judicial_decisions": """
        Whether politically sensitive rulings are implemented by the executive
        or ignored, undermined or defied. News coverage documents executive
        statements, policy reversals, or civil society responses to non-compliance
        with court rulings.
    """,
    "judicial_corruption_scandals": """
        Allegations, investigations or prosecutions of judicial actors for bribery,
        influence trading or abuse of office. Scandals involving judges, frequency
        and framing of corruption cases and their institutional consequences.
    """,
    "judicial_appointments_and_public_trust": """
        Transparency and contestation around judicial appointments, and public
        sentiment about judicial fairness and independence. Naming of judges,
        political salience of appointments, public confidence and trust in courts,
        opinion pieces and editorial framing of judicial impartiality.
    """,
}

THRESHOLDS = {
    "compliance_with_judicial_decisions": 0.5,
    "judicial_corruption_scandals": 0.5,
    "judicial_appointments_and_public_trust": 0.5,
}

CLASSIFIER_SYSTEM_PROMPT = """You are an expert classifier for de facto judicial independence events.
Classify whether a news event is related to judicial independence (1) or not (0).

Classify as 1 if the event involves:
- Courts or judges acting independently from political pressure
- Executive compliance or non-compliance with judicial decisions
- Judicial corruption scandals (bribery, influence trading by judicial actors)
- Judicial appointments, transparency, contestation, or politicization
- Public trust/distrust in judicial fairness and impartiality
- Political attacks on or interference with judicial institutions
- Judicial reform strengthening or weakening court independence
- Separation of powers conflicts involving the judiciary
- Prosecutors investigating government officials independently
- Constitutional courts striking down government actions

Classify as 0 if the event is about:
- General crime (drug busts, murders, robberies) not involving judicial independence
- Police operations, arrests, public safety
- Regulatory enforcement without court independence themes
- Elections, voter turnout without judicial involvement
- General government policy, infrastructure, education, healthcare

CRITICAL: Not every justice-related event is about judicial independence. A drug arrest is NOT
judicial independence. A court overturning a government decree IS.

Respond with ONLY 1 or 0. Nothing else."""

CLASSIFIER_USER_TEMPLATE = (
    "Country: {country} | Pillar: {pillar}\n\n"
    "Event:\n{event}\n\nClassification (1 or 0):"
)

SENTIMENT_SYSTEM_PROMPT = """You are a sentiment classifier for the World Justice Project (WJP). \
You will receive a news event that has already been identified as relevant to judicial independence. \
Your only task is to determine whether the event represents a threat to judicial independence or a \
strengthening of it.

## WHAT COUNTS AS A THREAT

The event describes an action, pattern, or situation that weakens the ability of courts and judges \
to operate independently. Examples:
- Politicized or non-transparent appointment or removal of judges.
- Executive or legislature ignoring, defying, or undermining court rulings.
- Political attacks, pressure campaigns, or retaliation against judges for their decisions.
- Laws that reduce judicial jurisdiction, pack courts, or curtail judicial review.
- Corruption, bribery, or influence trading involving judicial actors.
- Declining public trust or framing of courts as politically captured.
- Disciplinary proceedings used to punish judges for independent rulings.

## WHAT COUNTS AS A STRENGTHENING

The event describes an action, pattern, or situation that reinforces the ability of courts and \
judges to operate independently. Examples:
- Courts successfully blocking unlawful government actions or defending rights.
- Reinstatement of improperly suspended or dismissed judges.
- Anti-corruption investigations or accountability measures within the judiciary.
- Reforms that improve transparency in judicial appointments or tenure protections.
- Executive compliance with court rulings, especially on politically sensitive issues.
- Growing public confidence in judicial impartiality.
- International support, recognition, or conditionality reinforcing judicial autonomy.

## DECISION RULE

Ask: "Does this event make judicial independence stronger or weaker in practice?"
- Weaker → "threat"
- Stronger → "strengthening"
- Genuinely ambiguous or mixed → "neutral"

Respond with ONLY one word: threat, strengthening, or neutral. Nothing else."""

SENTIMENT_USER_TEMPLATE = (
    "Country: {country} | Pillar: {pillar}\n\n"
    "Event:\n{event}\n\nSentiment ('threat' or 'strengthening' or 'neutral'):"
)

# BERTopic hyperparameters
BERTOPIC_NR_TOPICS_GENERAL = 8
BERTOPIC_UMAP_NEIGHBORS = 15
BERTOPIC_UMAP_COMPONENTS = 10
BERTOPIC_HDBSCAN_MIN_CLUSTER = 5
BERTOPIC_HDBSCAN_MIN_SAMPLES = 3
BERTOPIC_CUSTOM_STOPS = [
    "article", "articles", "discussing", "discussed",
    "regarding", "concerning", "country", "report",
    "says", "said", "according", "also", "would",
    "could", "may", "might", "shall", "upon",
    "one", "two", "new", "made", "like",
    "including", "related", "based", "using",
    "italy", "italian", "polish", "poland", "hungary", "hungarian",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _skip(path, force: bool) -> bool:
    """Return True (and log) if the output already exists and force is False."""
    if path.exists() and not force:
        log.info("  Skipping %s (already exists — pass --force to re-run)", path.name)
        return True
    return False


def _save(df: pl.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    log.info("  Saved → %s", path)


# ── Module 1 ──────────────────────────────────────────────────────────────────

def run_m1_embeddings(df: pl.DataFrame, force: bool) -> None:
    out = PATH_DATA_INTERIM / "module1/df_embeddings_strategy_judicial_independence.parquet"
    if _skip(out, force):
        return
    log.info("Running embeddings strategy…")
    result = classify_events(
        df, "embeddings",
        model_name=EMBEDDING_MODEL,
        references=REFERENCES,
        thresholds=THRESHOLDS,
    )
    _save(result, out)


def run_m1_llm(df: pl.DataFrame, force: bool) -> None:
    out = PATH_DATA_INTERIM / "module1/df_llm_strategy_judicial_independence.parquet"
    if _skip(out, force):
        return
    log.info("Running local LLM strategy (%s)…", LOCAL_LLM_MODEL)
    result = classify_events(
        df, "llm",
        model_name=LOCAL_LLM_MODEL,
        system_prompt=CLASSIFIER_SYSTEM_PROMPT,
        user_msg_template=CLASSIFIER_USER_TEMPLATE,
    )
    _save(result, out)


def run_m1_llm_api(df: pl.DataFrame, client, model_name: str, force: bool) -> None:
    out = PATH_DATA_INTERIM / "module1/df_llm-api_strategy_judicial_independence.parquet"
    if _skip(out, force):
        return
    log.info("Running LLM API strategy (%s)…", model_name)
    result = classify_events(
        df, "llm-api",
        client=client,
        model_name=model_name,
        system_prompt=CLASSIFIER_SYSTEM_PROMPT,
        user_msg_template=CLASSIFIER_USER_TEMPLATE,
    )
    _save(result, out)


# ── Module 2: Sentiment ───────────────────────────────────────────────────────

def run_m2_sentiment(strategy: str, force: bool) -> None:
    out = PATH_DATA_INTERIM / f"module2/sentiment/df_m1_{strategy}_strategy_judicial_independence.parquet"
    if _skip(out, force):
        return
    log.info("Running sentiment classification (m1_strategy=%s)…", strategy)

    df = (
        pl.read_parquet(PATH_DATA_INTERIM / f"module1/df_{strategy}_strategy_judicial_independence.parquet")
        .filter(pl.col("is_judicial_independence"))
    )

    # Always join embeddings score_* columns so all strategies carry them
    embeddings_scores = PATH_DATA_INTERIM / "module1/df_embeddings_strategy_judicial_independence.parquet"
    if embeddings_scores.exists():
        df_scores = (
            pl.read_parquet(embeddings_scores)
            .select("country", "pillar", "event", cs.starts_with("score"))
        )
        df = df.join(df_scores, how="left", on=["country", "pillar", "event"])
    else:
        log.warning("  Embeddings scores not found — score_* columns will be missing.")

    result = classify_sentiment(
        df, "llm",
        model_name=LOCAL_LLM_MODEL,
        system_prompt=SENTIMENT_SYSTEM_PROMPT,
        user_msg_template=SENTIMENT_USER_TEMPLATE,
    )
    _save(result, out)


# ── Module 2: Topic Modeling ──────────────────────────────────────────────────

def run_m2_topics(strategy: str, force: bool) -> None:
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from bertopic.vectorizers import ClassTfidfTransformer

    df = pl.read_parquet(
        PATH_DATA_INTERIM / f"module2/sentiment/df_m1_{strategy}_strategy_judicial_independence.parquet"
    )

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    stop_words = list(ENGLISH_STOP_WORDS) + BERTOPIC_CUSTOM_STOPS

    def _make_bertopic(**kwargs) -> BERTopic:
        return BERTopic(
            embedding_model=embedding_model,
            umap_model=UMAP(
                n_neighbors=BERTOPIC_UMAP_NEIGHBORS,
                n_components=BERTOPIC_UMAP_COMPONENTS,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            ),
            hdbscan_model=HDBSCAN(
                min_cluster_size=BERTOPIC_HDBSCAN_MIN_CLUSTER,
                min_samples=BERTOPIC_HDBSCAN_MIN_SAMPLES,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            ),
            vectorizer_model=CountVectorizer(
                ngram_range=(1, 2),
                min_df=0.01,
                max_df=0.80,
                stop_words=stop_words,
                token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
            ),
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            **kwargs,
        )

    def _save_model(model: BERTopic, path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model.save(
            path,
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=EMBEDDING_MODEL,
        )
        log.info("  Saved → %s", path)

    # ── General model ─────────────────────────────────────────────────────────
    general_out = PATH_DATA_INTERIM / f"module2/topic_modelling/bertopic_general_m1_{strategy}"
    if not _skip(general_out, force):
        log.info("Fitting general BERTopic model (strategy=%s)…", strategy)
        model = _make_bertopic(
            representation_model=[
                KeyBERTInspired(top_n_words=15),
                MaximalMarginalRelevance(diversity=0.7),
            ],
            nr_topics=BERTOPIC_NR_TOPICS_GENERAL,
        )
        model.fit_transform(df["event"])
        _save_model(model, general_out)

    # ── Per-country models ────────────────────────────────────────────────────
    for (country,), df_country in df.group_by("country"):
        country_out = PATH_DATA_INTERIM / f"module2/topic_modelling/bertopic_{country}_m1_{strategy}"
        if _skip(country_out, force):
            continue
        log.info("Fitting BERTopic model for %s (strategy=%s)…", country, strategy)
        model = _make_bertopic(
            representation_model=[
                KeyBERTInspired(top_n_words=40),
                MaximalMarginalRelevance(diversity=0.6),
            ],
        )
        model.fit_transform(df_country["event"])
        _save_model(model, country_out)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WJP pipeline: Module 1 (classification) + Module 2 (sentiment + topics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        choices=["embeddings", "llm", "llm-api"],
        default=["embeddings", "llm"],
        metavar="STRATEGY",
        help="Module 1 strategies to run. Choices: embeddings llm llm-api",
    )
    p.add_argument(
        "--api-provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="Provider for the llm-api strategy",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key for llm-api strategy (falls back to OPENAI_API_KEY / ANTHROPIC_API_KEY env var)",
    )
    p.add_argument(
        "--api-model",
        default="gpt-4o-mini",
        help="Model name for llm-api strategy",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even if output files already exist",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log.info("Strategies: %s | Force: %s", args.strategies, args.force)

    # ── Raw data ──────────────────────────────────────────────────────────────
    log.info("Loading raw events from %s…", PATH_DATA_RAW)
    df = load_events(PATH_DATA_RAW)
    log.info("Loaded %d events across %d countries", len(df), df["country"].n_unique())

    # ── API client (llm-api only) ─────────────────────────────────────────────
    api_client = None
    if "llm-api" in args.strategies:
        env_var = "ANTHROPIC_API_KEY" if args.api_provider == "anthropic" else "OPENAI_API_KEY"
        key = args.api_key or os.environ.get(env_var)
        if not key:
            log.error("llm-api strategy requires --api-key or the %s environment variable.", env_var)
            sys.exit(1)
        if args.api_provider == "anthropic":
            import anthropic
            api_client = anthropic.Anthropic(api_key=key)
        else:
            import openai
            api_client = openai.OpenAI(api_key=key)
        log.info("API client ready (%s / %s)", args.api_provider, args.api_model)

    # ── Module 1 ──────────────────────────────────────────────────────────────
    log.info("=== MODULE 1: Classification ===")
    if "embeddings" in args.strategies:
        run_m1_embeddings(df, args.force)
    if "llm" in args.strategies:
        run_m1_llm(df, args.force)
    if "llm-api" in args.strategies:
        run_m1_llm_api(df, api_client, args.api_model, args.force)

    # ── Module 2: Sentiment ───────────────────────────────────────────────────
    log.info("=== MODULE 2: Sentiment ===")
    for strategy in args.strategies:
        run_m2_sentiment(strategy, args.force)

    # ── Module 2: Topic Modeling ──────────────────────────────────────────────
    log.info("=== MODULE 2: Topic Modeling ===")
    for strategy in args.strategies:
        run_m2_topics(strategy, args.force)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
