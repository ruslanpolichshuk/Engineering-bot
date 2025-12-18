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

# Railway deployment: use mounted volume path
# Railway volumes are typically mounted at /data or use RAILWAY_VOLUME_MOUNT_PATH
# For Railway: use /data/vectordb (volume mount) or /tmp/vectordb (temporary, cleared on deploy)
# For local: use ./vectordb
if os.getenv("RAILWAY_ENVIRONMENT"):
    # Check for Railway volume mount path (usually /data)
    volume_mount = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "/data")
    if os.path.exists(volume_mount) and os.access(volume_mount, os.W_OK):
        VECTORDB_DIR = os.getenv("VECTORDB_DIR", os.path.join(volume_mount, "vectordb"))
        print(f"[INFO] Using Railway volume at {VECTORDB_DIR}")
    else:
        # Fallback to /tmp if volume not available (will be cleared on deploy)
        VECTORDB_DIR = os.getenv("VECTORDB_DIR", "/tmp/vectordb")
        print(f"[WARN] Volume not available, using /tmp (data will be lost on deploy)")
else:
    # Local execution - use local directory
    VECTORDB_DIR = os.getenv("VECTORDB_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "vectordb"))

# Ensure the directory exists
os.makedirs(VECTORDB_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

if not API_KEY:
    raise ValueError("OPENAI_API_KEY не найден. Убедитесь, что он указан в .env или как переменная окружения.")
