import polars as pl

from wjp_judicial_independence.plot import plot_comparison


def compare_strategies(
    dfs: dict[str, pl.DataFrame], plot: bool = True
) -> dict[str, pl.DataFrame]:
    """Compare judicial independence classification results across strategies.

    Combines multiple classification DataFrames (one per strategy) and computes:

    - Overall JI rate per strategy
    - Per-pillar JI rate per strategy
    - Per-country JI rate per strategy
    - Pairwise agreement rates between strategies
    - Disagreement breakdown: events where only one strategy fires

    Optionally renders three plots: overall JI rates, per-pillar breakdown,
    and per-country breakdown.

    Args:
        dfs: Mapping of strategy name to its classified DataFrame. Each
            DataFrame must have ``country``, ``pillar``, ``impact``, ``event``,
            and ``is_judicial_independence`` columns, as produced by
            :func:`~wjp_judicial_independence.classifier.classify_events`.
        plot: If ``True`` (default), display comparison plots.

    Returns:
        Dictionary with the following keys:

        - ``"overall"`` — overall JI rate per strategy
        - ``"by_pillar"`` — JI rate per (pillar, strategy)
        - ``"by_country"`` — JI rate per (country, strategy)
        - ``"agreement"`` — pairwise agreement rates between strategies
        - ``"disagreement"`` — per-strategy count of unique True predictions
        - ``"combined"`` — row-aligned DataFrame with one boolean column per strategy
    """
    strategies = list(dfs.keys())

    # Build a combined DataFrame aligned by row
    base = dfs[strategies[0]].select(["country", "pillar", "impact", "event"])
    flags = [
        dfs[name].select(pl.col("is_judicial_independence").alias(name))
        for name in strategies
    ]
    combined = pl.concat([base, *flags], how="horizontal")

    # --- Overall JI rate ---
    overall = pl.DataFrame(
        {
            "strategy": strategies,
            "total_events": [len(combined)] * len(strategies),
            "ji_count": [int(combined[s].sum()) for s in strategies],
            "ji_rate": [float(combined[s].mean()) for s in strategies],
        }
    )

    # --- Per-pillar JI rate (tidy: pillar, strategy, ji_rate) ---
    pillar_rows = []
    for s in strategies:
        rates = (
            combined.group_by("pillar")
            .agg(pl.col(s).mean().alias("ji_rate"))
            .with_columns(pl.lit(s).alias("strategy"))
            .select(["pillar", "strategy", "ji_rate"])
        )
        pillar_rows.append(rates)
    by_pillar = pl.concat(pillar_rows).sort(["pillar", "strategy"])

    # --- Per-country JI rate (tidy: country, strategy, ji_rate) ---
    country_rows = []
    for s in strategies:
        rates = (
            combined.group_by("country")
            .agg(pl.col(s).mean().alias("ji_rate"))
            .with_columns(pl.lit(s).alias("strategy"))
            .select(["country", "strategy", "ji_rate"])
        )
        country_rows.append(rates)
    by_country = pl.concat(country_rows).sort(["country", "strategy"])

    # --- Pairwise agreement ---
    agreement_rows = []
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i + 1 :]:
            rate = (combined[s1] == combined[s2]).mean()
            agreement_rows.append(
                {"strategy_a": s1, "strategy_b": s2, "agreement_rate": rate}
            )
    agreement = pl.DataFrame(agreement_rows)

    # --- Disagreement: events where only one strategy fires True ---
    disagreement_rows = []
    for s in strategies:
        others = [o for o in strategies if o != s]
        only_this = combined[s]
        for o in others:
            only_this = only_this & ~combined[o]
        disagreement_rows.append(
            {"strategy": s, "unique_true_count": int(only_this.sum())}
        )

    all_true = combined[strategies[0]]
    for s in strategies[1:]:
        all_true = all_true & combined[s]
    disagreement_rows.append(
        {"strategy": "all_agree_true", "unique_true_count": int(all_true.sum())}
    )
    disagreement = pl.DataFrame(disagreement_rows)

    if plot:
        plot_comparison(overall, by_pillar, by_country, strategies)

    return {
        "overall": overall,
        "by_pillar": by_pillar,
        "by_country": by_country,
        "agreement": agreement,
        "disagreement": disagreement,
        "combined": combined,
    }
