# rag_assistant/config.py

import os
from dotenv import load_dotenv

load_dotenv()  # Загружает переменные из .env в окружение

API_KEY = os.getenv("OPENAI_API_KEY")
PDF_DIR = "downloaded_pdfs"
VECTORDB_DIR = "/vectordb"

if not API_KEY:
    raise ValueError("OPENAI_API_KEY не найден. Убедитесь, что он указан в .env или как переменная окружения.")
