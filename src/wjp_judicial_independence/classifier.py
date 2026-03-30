import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def classify_events(
    df: pl.DataFrame,
    model: SentenceTransformer,
    references: dict[str, str],
    thresholds: dict[str, float],
) -> pl.DataFrame:
    """Classify events by semantic similarity to judicial independence reference texts.

    Encodes each event and each reference description into embeddings, computes
    cosine similarity scores, and flags events that exceed any of the given
    thresholds as relevant to judicial independence.

    Args:
        df: DataFrame with at least an ``event`` column, as returned by
            :func:`~wjp_judicial_independence.preprocessing.load_events`.
        model: A loaded ``SentenceTransformer`` model used to encode texts.
        references: Mapping of category name to reference description text.
            Keys become the score column names (prefixed with ``score_``).
        thresholds: Mapping of category name to similarity threshold.
            Must contain the same keys as *references*.

    Returns:
        Input DataFrame extended with one ``score_<category>`` column per
        reference and a boolean ``is_judicial_independence`` column that is
        ``True`` when any score exceeds its threshold.
    """
    event_embeddings = model.encode(df["event"])

    ref_items = list(references.items())
    reference_embeddings = model.encode([text for _, text in ref_items])

    scores = cosine_similarity(event_embeddings, reference_embeddings)
    score_columns = {f"score_{key}": pl.Float32 for key, _ in ref_items}
    df_scores = pl.DataFrame(scores, schema=score_columns)

    df = pl.concat([df.with_columns(embedding = event_embeddings), df_scores], how="horizontal")

    is_judicial_independence = pl.lit(False)
    for category, threshold in thresholds.items():
        is_judicial_independence = is_judicial_independence | (
            pl.col(f"score_{category}") > threshold
        )

    return df.with_columns(is_judicial_independence=is_judicial_independence)
