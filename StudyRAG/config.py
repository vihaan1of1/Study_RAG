"""StudyRAG Application Configuration Module.

Handles environment loading, project path resolution, model defaults, and runtime directories.
"""

from __future__ import annotations

import os
# Prevent OpenMP runtime collision on Windows between PyTorch and FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from dotenv import load_dotenv

# Resolve project root (directory containing this config.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# Load environment variables from .env if present
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()  # Fallback to system env

# Runtime storage directories
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
INDEXES_DIR = DATA_DIR / "indexes"
ASSETS_DIR = PROJECT_ROOT / "assets"

for d in [DATA_DIR, UPLOADS_DIR, INDEXES_DIR, ASSETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Provider Configuration
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
SUPPORTED_PROVIDERS = ["gemini", "groq", "offline"]

# Google Gemini Configuration
DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
SUPPORTED_GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
]

# Groq Configuration
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
SUPPORTED_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
]

# Vector & Embedding Configuration
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
).strip()
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DEFAULT_TOP_K = int(os.getenv("TOP_K_RETRIEVAL", "5"))

# Application Branding
APP_TITLE = "StudyRAG"
APP_TAGLINE = "Your personal AI-powered study companion"
APP_ICON = "📚"


def _read_env_or_secret(key: str, default: str = "") -> str:
    """Read a setting from OS env var, falling back to Streamlit Cloud secrets."""
    val = os.getenv(key, "")
    if not val:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and key in st.secrets:
                val = str(st.secrets[key])
        except Exception:
            pass
    return val.strip() if val else default


def get_gemini_api_key(custom_key: str | None = None) -> str:
    """Retrieve effective Gemini API key (custom override > env var > st.secrets)."""
    if custom_key and custom_key.strip():
        return custom_key.strip()
    return _read_env_or_secret("GEMINI_API_KEY", DEFAULT_GEMINI_API_KEY)


def is_gemini_configured(custom_key: str | None = None) -> bool:
    """Check if a valid Gemini API key is available."""
    key = get_gemini_api_key(custom_key)
    return bool(key and key != "your_gemini_api_key_here")


def get_groq_api_key(custom_key: str | None = None) -> str:
    """Retrieve effective Groq API key (custom override > env var > st.secrets)."""
    if custom_key and custom_key.strip():
        return custom_key.strip()
    return _read_env_or_secret("GROQ_API_KEY", DEFAULT_GROQ_API_KEY)


def is_groq_configured(custom_key: str | None = None) -> bool:
    """Check if a non-empty Groq API key is available."""
    key = get_groq_api_key(custom_key)
    return bool(key and key != "your_groq_api_key_here")


def is_llm_configured(provider: str, custom_key: str | None = None) -> bool:
    """Check if the selected provider is configured."""
    p = (provider or "").strip().lower()
    if p == "gemini":
        return is_gemini_configured(custom_key)
    if p == "groq":
        return is_groq_configured(custom_key)
    if p == "offline":
        return True
    return False
