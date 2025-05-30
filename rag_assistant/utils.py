import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

import os
import time
import shutil
import pdfplumber
from tqdm import tqdm
from retrying import retry
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pdfminer.pdfparser import PDFSyntaxError
from rag_assistant import config

def parse_pdfs(pdf_dir: str) -> list[Document]:
    docs: list[Document] = []
    all_files = os.listdir(pdf_dir)
    pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]

    print(f"[INFO] Всего PDF-файлов к обработке: {len(pdf_files)}")

    for fname in pdf_files:
        path = os.path.join(pdf_dir, fname)
        try:
            print(f"[LOAD] Загружаем PDF: {fname}")
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text or len(text.strip()) < 20:
                        print(f"[WARN] Пустая или слишком короткая страница {i+1} в {fname}")
                        continue
                    meta = {'source': fname, 'page': i + 1}
                    docs.append(Document(page_content=text, metadata=meta))
        except PDFSyntaxError:
            print(f"[ERROR] Повреждён PDF: {fname}")
        except Exception as e:
            print(f"[ERROR] Ошибка с {fname}: {e}")

    print(f"[RESULT] Загружено {len(docs)} страниц из {len(pdf_files)} PDF-файлов")
    return docs


@retry(stop_max_attempt_number=3, wait_fixed=1000)
def get_or_create_vectorstore_incremental(pdf_dir, persist_dir):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        batch_size=64,
        request_timeout=60,
        show_progress_bar=True
    )

    os.makedirs(persist_dir, exist_ok=True)

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    existing_files = set()
    try:
        existing_metadatas = vectordb.get()["metadatas"]
        for m in existing_metadatas:
            if isinstance(m, dict) and "source" in m:
                existing_files.add(m["source"])
    except Exception as e:
        print(f"[WARN] Не удалось получить список уже добавленных файлов: {e}")

    print(f"[INFO] В базе уже есть {len(existing_files)} документов.")

    new_pdfs = [
        f for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf") and f not in existing_files
    ]

    if not new_pdfs:
        print("[INFO] Новых PDF-файлов не найдено. База не обновлена.")
        return vectordb

    print(f"[INFO] Найдено новых PDF: {len(new_pdfs)}")

    docs: list[Document] = []
    for fname in new_pdfs:
        try:
            path = os.path.join(pdf_dir, fname)
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text or len(text.strip()) < 20:
                        continue
                    meta = {'source': fname, 'page': i + 1}
                    docs.append(Document(page_content=text, metadata=meta))
        except Exception as e:
            print(f"[ERROR] Ошибка при чтении {fname}: {e}")

    print(f"[INFO] Всего новых страниц: {len(docs)}")

    if not docs:
        print("[WARN] Нет валидных страниц для добавления.")
        return vectordb

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Чанков для добавления: {len(chunks)}")

    batch_size = 64
    for i in tqdm(range(0, len(chunks), batch_size), desc="📥 Добавление новых чанков"):
        batch = chunks[i:i + batch_size]
        try:
            vectordb.add_documents(batch)
            vectordb.persist()
            time.sleep(0.5)
        except Exception as e:
            print(f"[ERROR] Ошибка при добавлении батча {i // batch_size + 1}: {e}")

    print("[SUCCESS] Новые документы успешно добавлены в базу.")
    return vectordb
