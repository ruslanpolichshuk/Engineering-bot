# rag_assistant/config.py

import os
from dotenv import load_dotenv

load_dotenv()  # Загружает переменные из .env в окружение

API_KEY = os.getenv("OPENAI_API_KEY")
PDF_DIR = os.getenv("PDF_DIR", "downloaded_pdfs")

# OpenAI Model Configuration
# GPT-5.2-Pro is the latest model (released Dec 2025) with 400k context window
# Falls back to GPT-4o if GPT-5.2-Pro is not available
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2-pro")

# Vector database directory configuration
# For Railway: use /vectordb (persistent directory, survives deploys)
# For local: use ./vectordb
if os.getenv("RAILWAY_ENVIRONMENT"):
    # Use /vectordb as default for Railway (persistent, not temporary)
    # Can be overridden via VECTORDB_DIR environment variable
    VECTORDB_DIR = os.getenv("VECTORDB_DIR", "/vectordb")
    print(f"[INFO] Using persistent vector database directory: {VECTORDB_DIR}")
else:
    # Local execution - use local directory
    VECTORDB_DIR = os.getenv("VECTORDB_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "vectordb"))

# Ensure the directory exists
os.makedirs(VECTORDB_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

if not API_KEY:
    raise ValueError("OPENAI_API_KEY не найден. Убедитесь, что он указан в .env или как переменная окружения.")
