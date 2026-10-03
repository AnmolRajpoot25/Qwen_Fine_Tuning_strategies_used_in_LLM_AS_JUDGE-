
from pathlib import Path


PROJECT_ROOT = Path(
    "/content/drive/MyDrive/LLM-Judge"
)


RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)


PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


VALIDATION_RATIO = 0.10


RANDOM_SEED = 42


SUPPORTED_CATEGORIES = [

    "DSA",

    "Competitive Programming",

    "C++ / Python",

    "SQL / DBMS",

    "OS / Systems",

    "Backend / System Design",

    "AI / ML",

    "LLM / RAG / Agents",

    "Frontend",

    "DevOps / Cloud"
]
