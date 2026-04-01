import json
from pathlib import Path

import polars as pl

PILLARS_MAPPER = {
    "Pillar 1": "Constraints on Government Powers",
    "Pillar 2": "Absence of Corruption",
    "Pillar 3": "Open Government",
    "Pillar 4": "Fundamental Freedoms",
    "Pillar 5": "Order and Security",
    "Pillar 6": "Regulatory Enforcement",
    "Pillar 7": "Civil Justice",
    "Pillar 8": "Criminal Justice",
}


def _validate_country_json(data: object, path: Path) -> None:
    """Validate that *data* matches the expected country JSON structure.

    Expected shape::

        {
            "Pillar N": {
                "<impact category>": "<summary string>",
                ...
            },
            ...
        }

    Args:
        data: Parsed JSON value to validate.
        path: Source file path, used only for error messages.

    Raises:
        ValueError: If the root value is not a dict, if any pillar key is not
            in ``PILLARS_MAPPER``, if a pillar value is not a dict, or if an
            impact summary is not a string.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected a JSON object at root, got {type(data).__name__}"
        )

    unknown_pillars = set(data.keys()) - PILLARS_MAPPER.keys()
    if unknown_pillars:
        raise ValueError(f"{path}: unknown pillar keys: {sorted(unknown_pillars)}")

    for pillar, impacts in data.items():
        if not isinstance(impacts, dict):
            raise ValueError(
                f"{path}: expected a JSON object for '{pillar}', got {type(impacts).__name__}"
            )
        for impact, summary in impacts.items():
            if not isinstance(summary, str):
                raise ValueError(
                    f"{path}: expected a string for '{pillar}' -> '{impact}', "
                    f"got {type(summary).__name__}"
                )


def load_events(folder_path: Path | str) -> pl.DataFrame:
    """Load and preprocess all country JSON files in *folder_path*.

    Recursively finds every ``*.json`` file under *folder_path*, parses it,
    validates its structure with :func:`_validate_country_json`, and extracts
    individual news events from the nested pillar/impact summaries.

    Each summary is split on double newlines (``\\n\\n``) and only paragraphs
    that start with ``* **`` are kept, filtering out conclusions, titles, and
    other non-event content.

    Args:
        folder_path: Path to the directory containing country JSON files.

    Returns:
        DataFrame with columns ``country``, ``pillar``, ``impact``, and
        ``event``, one row per news event extracted from the summaries.

    Raises:
        ValueError: If any JSON file cannot be parsed or fails structure
            validation.
    """
    rows = []
    for path in Path(folder_path).rglob("*.json"):
        with open(path) as f:
            country_news = json.load(f)

        _validate_country_json(country_news, path)

        country_name = path.stem.upper()
        for pillar, impacts in country_news.items():
            for impact, summary in impacts.items():
                rows.append([country_name, PILLARS_MAPPER[pillar], impact, summary])

    df = pl.DataFrame(
        rows,
        schema={
            "country": pl.String,
            "pillar": pl.String,
            "impact": pl.String,
            "summary_news": pl.String,
        },
        orient="row",
    )
    # Crear dataframe por eventos
    df = (
        df
        # Hacemos chunking por parrafos. un nuevo parrafo es \n\n
        .with_columns(paragraph=pl.col("summary_news").str.split("\n"))
        .explode("paragraph")
        .with_columns(paragraph=pl.col("paragraph").str.strip_chars())
        # Un párrafo puede contener la conclusion, un resumen, titulos, etc. Filtramos para que solo se utilicen los párrafos que son eventos!
        .filter(pl.col("paragraph").str.starts_with("* **"))
        .select("country", "pillar", "impact", event="paragraph")
    )
    return df
