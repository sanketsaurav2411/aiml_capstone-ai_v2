import os
from dotenv import load_dotenv

# Load environment file
load_dotenv()

APP_TITLE = os.getenv("APP_TITLE", "RAG API")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
APP_ENV = os.getenv("APP_ENV", "development")
APP_PORT = int(os.getenv("APP_PORT", "8001"))
VECTOR_DB_BASE_URL = os.getenv("VECTOR_DB_BASE_URL", "http://localhost:8000")
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
