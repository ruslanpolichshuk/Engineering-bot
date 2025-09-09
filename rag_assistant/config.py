# rag_assistant/config.py

import os
from dotenv import load_dotenv

load_dotenv()  # Загружает переменные из .env в окружение

API_KEY = os.getenv("OPENAI_API_KEY")
PDF_DIR = "downloaded_pdfs"
VECTORDB_DIR = "/vectordb"

if not API_KEY:
    raise ValueError("OPENAI_API_KEY не найден. Убедитесь, что он указан в .env или как переменная окружения.")

# =========================
# Self-RAG configuration
# =========================

# Base model and decoding
SELF_RAG_MODEL = os.getenv("SELF_RAG_MODEL", "gpt-4o-mini")
SELF_RAG_TEMPERATURE = float(os.getenv("SELF_RAG_TEMPERATURE", "0"))

# Retrieval parameters
SELF_RAG_SEARCH_K = int(os.getenv("SELF_RAG_SEARCH_K", "10"))
SELF_RAG_FETCH_K = int(os.getenv("SELF_RAG_FETCH_K", "30"))
SELF_RAG_SCORE_THRESHOLD = float(os.getenv("SELF_RAG_SCORE_THRESHOLD", "0.4"))
SELF_RAG_MAX_CONTEXT_CHUNKS = int(os.getenv("SELF_RAG_MAX_CONTEXT_CHUNKS", "10"))

# Iterative refinement
SELF_RAG_MAX_ROUNDS = int(os.getenv("SELF_RAG_MAX_ROUNDS", "2"))
