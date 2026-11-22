import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BASE_DIR.parent

APP_TITLE = os.getenv("APP_TITLE", "TariffFlow AI - Indian Customs & Tariff Intelligence Service")
APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
APP_ENV = os.getenv("APP_ENV", "development")
APP_PORT = int(os.getenv("APP_PORT", "8001"))

VECTOR_DB_BASE_URL = os.getenv("VECTOR_DB_BASE_URL", "")
GCS_CHROMA_SNAPSHOT_URI = os.getenv("GCS_CHROMA_SNAPSHOT_URI", "")
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "itc_tariff_items")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))


def resolve_chroma_persist_dir() -> str:
    explicit = os.getenv("CHROMA_PERSIST_DIR", "").strip()
    if explicit:
        return explicit

    candidates = [
        BASE_DIR / "data" / "chroma_db_v0_3",
        WORKSPACE_ROOT / "02_processed_data" / "chroma_db_v0_3",
        WORKSPACE_ROOT / "02_processed_data" / "chroma_db_base_64_v0.73",
        WORKSPACE_ROOT / "02_processed_data" / "chroma_db_large_64_v0.74",
        WORKSPACE_ROOT / "02_processed_data" / "chroma_db_small_256_v0.71",
        BASE_DIR / "data" / "chroma_db",
    ]
    for path in candidates:
        if (path / "chroma.sqlite3").exists():
            return str(path)
    return str(BASE_DIR / "data" / "chroma_db")


CHROMA_PERSIST_DIR = resolve_chroma_persist_dir()


def resolve_embed_model_name(persist_dir: str) -> str:
    explicit = os.getenv("EMBED_MODEL_NAME", "").strip()
    if explicit:
        return explicit
    lower = persist_dir.lower()
    if "small" in lower:
        return "BAAI/bge-small-en-v1.5"
    if "large" in lower:
        return "BAAI/bge-large-en-v1.5"
    if "m3" in lower:
        return "BAAI/bge-m3"
    return "BAAI/bge-base-en-v1.5"


EMBED_MODEL_NAME = resolve_embed_model_name(CHROMA_PERSIST_DIR)
ENABLE_LOCAL_EMBEDDINGS = os.getenv("ENABLE_LOCAL_EMBEDDINGS", "false").lower() == "true"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FAST_MODEL = os.getenv("GEMINI_FAST_MODEL", GEMINI_MODEL)

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "data" / "ai_sessions.sqlite"))
DATABASE_URL = os.getenv("DATABASE_URL", "")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:4000,http://localhost:5173").split(",")
    if origin.strip()
]

TARIFF_VERSION = os.getenv(
    "TARIFF_VERSION",
    "ITC-HS 2022 (DGCI&S Schedule I, effective 1 April 2022)",
)
ABSTENTION_CONFIDENCE_THRESHOLD = float(os.getenv("ABSTENTION_CONFIDENCE_THRESHOLD", "0.65"))
ABSTENTION_SIMILARITY_THRESHOLD = float(os.getenv("ABSTENTION_SIMILARITY_THRESHOLD", "0.20"))
