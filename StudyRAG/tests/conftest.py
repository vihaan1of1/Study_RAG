"""Pytest configuration and environment fixtures for StudyRAG."""

import os
import sys
from pathlib import Path

# Fix OpenMP duplicate runtime conflicts on Windows between PyTorch and FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add project root to sys.path so imports like `src.xyz` work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pre-import faiss before other modules
try:
    import faiss
except ImportError:
    pass
