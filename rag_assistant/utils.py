import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import os
import time
import shutil
from tqdm import tqdm
from retrying import retry
import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pdfminer.pdfparser import PDFSyntaxError
from rag_assistant import config


def parse_pdfs(pdf_dir: str) -> list[Document]:
    docs: list[Document] = []
    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith('.pdf'):
            continue
        path = os.path.join(pdf_dir, fname)
        try:
            print(f"[LOAD] Загружаем PDF: {fname}")
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text:
                        print(f"[WARN] Пустая страница {i+1} в {fname}")
                        continue
                    meta = {'source': fname, 'page': i + 1}
                    docs.append(Document(page_content=text, metadata=meta))
        except PDFSyntaxError:
            print(f"[ERROR] Повреждён PDF: {fname}")
        except Exception as e:
            print(f"[ERROR] Ошибка с {fname}: {e}")
    print(f"[RESULT] Загружено {len(docs)} страниц из {len(os.listdir(pdf_dir))} PDF-файлов")
    return docs


@retry(stop_max_attempt_number=3, wait_fixed=1000)
from tqdm import tqdm
import time
import shutil
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from rag_assistant.utils import parse_pdfs
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_or_create_vectorstore(pdf_dir, persist_dir, force_rebuild=False):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        batch_size=64,           # 🔧 большой батч — меньше запросов
        request_timeout=60,
        show_progress_bar=True
    )

    if not force_rebuild:
        try:
            vectordb = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
            if vectordb._collection.count() > 0:
                print("[INFO] Загружена существующая база векторов.")
                return vectordb
        except Exception as e:
            print(f"[WARN] Ошибка загрузки базы: {e}")

    # Удалим старую базу, если нужно
    if force_rebuild and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
    os.makedirs(persist_dir, exist_ok=True)

    # Шаг 1: загрузка и разбивка PDF
    docs = parse_pdfs(pdf_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Разбито на {len(chunks)} чанков")

    # Шаг 2: создаём пустую базу
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # Шаг 3: загружаем батчами
    batch_size = 64
    for i in tqdm(range(0, len(chunks), batch_size), desc="⬆️ Загрузка в Chroma"):
        chunk_batch = chunks[i:i + batch_size]
        try:
            vectordb.add_documents(chunk_batch)
            vectordb.persist()
            time.sleep(0.5)  # защита от перегрузки API
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке батча {i // batch_size + 1}: {e}")

    print("[SUCCESS] Все чанки успешно сохранены в векторную базу.")
    return vectordb
