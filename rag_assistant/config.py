# rag_assistant/config.py

import os
from dotenv import load_dotenv

load_dotenv()  # Загружает переменные из .env в окружение

API_KEY = os.getenv("OPENAI_API_KEY")
PDF_DIR = os.getenv("PDF_DIR", "downloaded_pdfs")
VECTORDB_DIR = os.path.join(os.getcwd(), os.getenv("VECTORDB_DIR", "vectordb"))

if not API_KEY:
    raise ValueError("OPENAI_API_KEY не найден. Убедитесь, что он указан в .env или как переменная окружения.")

# =========================
# Knowledge Graph Configuration
# =========================
RECREATE = os.getenv("RECREATE", "0") == "1"
KNOWLEDGE_GRAPH_DIR = os.path.join(os.getcwd(), os.getenv("KNOWLEDGE_GRAPH_DIR", "knowledge_graph"))
USE_KNOWLEDGE_GRAPH = os.getenv("USE_KNOWLEDGE_GRAPH", "true").lower() == "true"

# Knowledge Graph Parameters
KG_ENTITY_EXTRACTION_MODEL = os.getenv("KG_ENTITY_EXTRACTION_MODEL", "gpt-5")
KG_RELATION_EXTRACTION_MODEL = os.getenv("KG_RELATION_EXTRACTION_MODEL", "gpt-5")
KG_MAX_ENTITIES_PER_CHUNK = int(os.getenv("KG_MAX_ENTITIES_PER_CHUNK", "20"))
KG_MAX_RELATIONS_PER_CHUNK = int(os.getenv("KG_MAX_RELATIONS_PER_CHUNK", "15"))

# =========================
# Self-RAG configuration
# =========================

# Base model and decoding - Updated to GPT-5
SELF_RAG_MODEL = os.getenv("SELF_RAG_MODEL", "gpt-5")
SELF_RAG_TEMPERATURE = float(os.getenv("SELF_RAG_TEMPERATURE", "0"))

# Retrieval parameters
SELF_RAG_SEARCH_K = int(os.getenv("SELF_RAG_SEARCH_K", "10"))
SELF_RAG_FETCH_K = int(os.getenv("SELF_RAG_FETCH_K", "30"))
SELF_RAG_SCORE_THRESHOLD = float(os.getenv("SELF_RAG_SCORE_THRESHOLD", "0.4"))
SELF_RAG_MAX_CONTEXT_CHUNKS = int(os.getenv("SELF_RAG_MAX_CONTEXT_CHUNKS", "10"))

# Iterative refinement
SELF_RAG_MAX_ROUNDS = int(os.getenv("SELF_RAG_MAX_ROUNDS", "2"))
