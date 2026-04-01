from typing import Any, Literal

import polars as pl
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from wjp_judicial_independence.utils import call_api

LABELS = ("threat", "strengthening", "neutral")


def classify_sentiment(
    df: pl.DataFrame,
    method: Literal["embeddings", "llm", "llm-api"],
    **kwargs,
) -> pl.DataFrame:
    """Classify events as a threat, strengthening, or neutral to judicial independence.

    Dispatches to one of three classification strategies based on *method*.

    **Embeddings** (``method="embeddings"``) — cosine similarity between sentence
    embeddings and reference category descriptions. Required kwargs:

    - ``model_name`` (``str``) — HuggingFace model ID for ``SentenceTransformer``
    - ``references`` (``dict[str, str]``) — category name → description text.
      Keys must include ``"threat"``, ``"strengthening"``, and ``"neutral"``.
    - ``thresholds`` (``dict[str, float]``) — category name → similarity threshold.
      Must contain the same keys as *references*.

    **LLM** (``method="llm"``) — loads a 4-bit quantized local causal LM. Required kwargs:

    - ``model_name`` (``str``) — HuggingFace model ID.
    - ``system_prompt`` (``str``, optional) — defaults to :data:`DEFAULT_SYSTEM_PROMPT`.
    - ``user_msg_template`` (``str``, optional) — defaults to
      :data:`DEFAULT_USER_MSG_TEMPLATE`.

    **API** (``method="llm-api"``) — calls a paid API provider (Anthropic or OpenAI).
    Required kwargs:

    - ``client`` — initialized client, e.g. ``anthropic.Anthropic()`` or
      ``openai.OpenAI()``.
    - ``model_name`` (``str``) — model ID, e.g. ``"claude-opus-4-6"`` or ``"gpt-4o"``.
    - ``system_prompt`` (``str``, optional) — defaults to :data:`DEFAULT_SYSTEM_PROMPT`.
    - ``user_msg_template`` (``str``, optional) — defaults to
      :data:`DEFAULT_USER_MSG_TEMPLATE`.

    Args:
        df: DataFrame with at least an ``event`` column, as returned by
            :func:`~wjp_judicial_independence.preprocessing.load_events`.
        method: Classification strategy — ``"embeddings"``, ``"llm"``, or ``"llm-api"``.
        **kwargs: Method-specific arguments as described above.

    Returns:
        Input DataFrame extended with a ``judicial_independence_sentiment`` column
        whose values are one of ``"threat"``, ``"strengthening"``, or ``"neutral"``
        (and embedding/score columns when using ``"embeddings"``).

    Raises:
        ValueError: If *method* is not recognised or the API client provider is
            not supported.
    """
    if method == "embeddings":
        return _classify_with_embeddings(df, **kwargs)
    elif method == "llm":
        return _classify_with_llm(df, **kwargs)
    elif method == "llm-api":
        return _classify_with_api(df, **kwargs)
    else:
        raise ValueError(
            f"Unknown method {method!r}. Expected 'embeddings', 'llm', or 'llm-api'."
        )


def _classify_with_embeddings(
    df: pl.DataFrame,
    model_name: str,
    references: dict[str, str],
    thresholds: dict[str, float],
) -> pl.DataFrame:
    """Classify events using cosine similarity between sentence embeddings.

    Encodes each event and each reference description into embeddings, computes
    cosine similarity scores, then assigns the highest-scoring category whose
    score exceeds its threshold. Defaults to ``"neutral"`` when no threshold
    is exceeded.

    Args:
        df: DataFrame with at least an ``event`` column.
        model_name: HuggingFace model ID passed to ``SentenceTransformer``.
        references: Mapping of category name to reference description text.
            Keys become score column names (prefixed with ``score_``).
        thresholds: Mapping of category name to similarity threshold.
            Must contain the same keys as *references*.

    Returns:
        Input DataFrame extended with the event ``embedding``, one
        ``score_<category>`` column per reference, and a
        ``judicial_independence_sentiment`` column.
    """
    model = SentenceTransformer(model_name)
    event_embeddings = model.encode(df["event"])

    ref_items = list(references.items())
    reference_embeddings = model.encode([text for _, text in ref_items])

    scores = cosine_similarity(event_embeddings, reference_embeddings)
    score_columns = {f"score_{key}": pl.Float32 for key, _ in ref_items}
    df_scores = pl.DataFrame(scores, schema=score_columns)

    df = pl.concat(
        [df.with_columns(embedding=event_embeddings), df_scores], how="horizontal"
    )

    categories = [key for key, _ in ref_items]
    sentiments = []
    for row in df_scores.iter_rows(named=True):
        best_label = "neutral"
        best_score = -1.0
        for category in categories:
            score = row[f"score_{category}"]
            threshold = thresholds.get(category, 1.0)
            if score > threshold and score > best_score:
                best_score = score
                best_label = category
        sentiments.append(best_label)

    return df.with_columns(judicial_independence_sentiment=pl.Series(sentiments))


def _classify_with_llm(
    df: pl.DataFrame,
    model_name: str,
    system_prompt: str,
    user_msg_template: str,
) -> pl.DataFrame:
    """Classify events using a quantized causal LLM prompted for three-class classification.

    For each event, a chat-formatted prompt is built with the country and pillar
    as context and the event text as input. The model is expected to respond with
    ``"threat"``, ``"strengthening"``, or ``"neutral"``.

    Args:
        df: DataFrame with at least ``event``, ``country``, and ``pillar`` columns.
        model_name: HuggingFace model ID to load in 4-bit quantization via
            ``BitsAndBytesConfig``.
        system_prompt: System message that instructs the model how to classify.
            Defaults to :data:`DEFAULT_SYSTEM_PROMPT`.
        user_msg_template: Template string for the user message, formatted with
            row fields via ``str.format_map``. Defaults to
            :data:`DEFAULT_USER_MSG_TEMPLATE`.

    Returns:
        Input DataFrame extended with a ``judicial_independence_sentiment`` column.
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    results = []

    for row in tqdm(df.iter_rows(named=True), total=len(df)):
        user_msg = user_msg_template.format_map(row)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
            )

        response = (
            tokenizer.decode(
                output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            .strip()
            .lower()
        )

        results.append(_parse_label(response))

    return df.with_columns(judicial_independence_sentiment=pl.Series(results))


def _classify_with_api(
    df: pl.DataFrame,
    client: Any,
    model_name: str,
    system_prompt: str,
    user_msg_template: str,
) -> pl.DataFrame:
    """Classify events by calling a paid API (Anthropic or OpenAI).

    For each event, a chat prompt is sent to the API and the response is parsed
    as one of ``"threat"``, ``"strengthening"``, or ``"neutral"``.

    Args:
        df: DataFrame with at least ``event``, ``country``, and ``pillar`` columns.
        client: Initialized API client — ``anthropic.Anthropic()`` or
            ``openai.OpenAI()``. Provider is detected automatically.
        model_name: Model ID to use (e.g. ``"claude-opus-4-6"`` or ``"gpt-4o"``).
        system_prompt: System message that instructs the model how to classify.
            Defaults to :data:`DEFAULT_SYSTEM_PROMPT`.
        user_msg_template: Template string for the user message, formatted with
            row fields via ``str.format_map``. Defaults to
            :data:`DEFAULT_USER_MSG_TEMPLATE`.

    Returns:
        Input DataFrame extended with a ``judicial_independence_sentiment`` column.
    """
    results = []

    for row in tqdm(df.iter_rows(named=True), total=len(df)):
        user_msg = user_msg_template.format_map(row)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        response = call_api(client, model_name, messages)
        results.append(_parse_label(response.lower()))

    return df.with_columns(judicial_independence_sentiment=pl.Series(results))


def _parse_label(response: str) -> str:
    """Extract a sentiment label from a model response string.

    Checks whether any of the known labels (``"threat"``, ``"strengthening"``,
    ``"neutral"``) appear in the response and returns the first match.
    Falls back to ``"neutral"`` if none are found.

    Args:
        response: Raw lowercased text returned by the model.

    Returns:
        One of ``"threat"``, ``"strengthening"``, or ``"neutral"``.
    """
    for label in LABELS:
        if label in response:
            return label
    return "neutral"
