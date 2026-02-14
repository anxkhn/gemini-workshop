"""Backend configuration. Edit this file to change defaults."""

import os

# Disable ChromaDB telemetry (avoids posthog connection errors)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# --- Server ---
HOST = "127.0.0.1"
PORT = 8000

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
SAMPLE_DOCS_DIR = os.path.join(BASE_DIR, "sample_docs")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# --- RAG defaults ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
RAG_TOP_K = 5
COLLECTION_NAME = "workshop_docs"

# --- Generation defaults ---
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40
DEFAULT_MAX_TOKENS = 2048

# --- Model card links (by family) ---
MODEL_CARD_BASE = "https://ai.google.dev/gemini-api/docs/models"
MODEL_CARDS = {
    "gemini-2.5": "https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash",
    "gemini-2.0": "https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash",
    "gemini-1.5": "https://ai.google.dev/gemini-api/docs/models#gemini-1.5-flash",
    "gemini-1.0": "https://ai.google.dev/gemini-api/docs/models#gemini-1.0-pro",
    "gemini-embedding": "https://ai.google.dev/gemini-api/docs/models#gemini-embedding",
    "imagen": "https://ai.google.dev/gemini-api/docs/models#imagen-3",
}

# --- Retry ---
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# --- Safety ---
# Never log or store API keys
REDACT_KEY_IN_LOGS = True

# Ensure upload dir exists
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
