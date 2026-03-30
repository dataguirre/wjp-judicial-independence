from pathlib import Path

"""
DATA PATHS
"""

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PATH_DATA_RAW = ROOT_DIR / "data/raw"
PATH_DATA_INTERIM = ROOT_DIR / "data/interim"
PATH_DATA_PROCESSED = ROOT_DIR / "data/processed"
