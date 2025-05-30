import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

import os
import time
import shutil
import gc
import pdfplumber
from tqdm import tqdm
from retrying import retry
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pdfminer.pdfparser import PDFSyntaxError
from rag_assistant import config

def parse_pdf_file(path: str, fname: str) -> list[Document]:
    docs = []
    try:
        with pdfplumber.open(path) as pdf:
            print(f"[DEBUG] {fname}: {len(pdf.pages)} стр")
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text or len(text.strip()) < 20:
                    continue
                meta = {'source': fname, 'page': i + 1}
                docs.append(Document(page_content=text, metadata=meta))
    except PDFSyntaxError:
        print(f"[ERROR] Повреждён PDF: {fname}")
    except Exception as e:
        print(f"[ERROR] Ошибка с {fname}: {e}")
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

    # Определяем уже загруженные документы
    existing_files = set()
    try:
        for m in vectordb.get()["metadatas"]:
            if isinstance(m, dict) and "source" in m:
                existing_files.add(m["source"])
    except Exception as e:
        print(f"[WARN] Не удалось получить список загруженных PDF: {e}")

    all_pdfs = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    new_pdfs = [f for f in all_pdfs if f not in existing_files]
    print(f"[INFO] Всего новых PDF: {len(new_pdfs)}")

    if not new_pdfs:
        print("[INFO] Новых PDF не найдено. Пропускаем обновление базы.")
        return vectordb

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    batch_size = 8
    total_docs = 0
    total_chunks = 0

    for i in range(0, len(new_pdfs), batch_size):
        batch_files = new_pdfs[i:i + batch_size]
        docs: list[Document] = []

        for fname in batch_files:
            path = os.path.join(pdf_dir, fname)
            print(f"[LOAD] {fname}")
            docs.extend(parse_pdf_file(path, fname))

        print(f"[INFO] Загружено страниц из батча: {len(docs)}")
        total_docs += len(docs)

        if not docs:
            continue

        chunks = splitter.split_documents(docs)
        print(f"[INFO] Чанков из батча: {len(chunks)}")
        total_chunks += len(chunks)

        for j in tqdm(range(0, len(chunks), 64), desc=f"📥 Добавление батча {i // batch_size + 1}"):
            chunk_batch = chunks[j:j + 64]
            try:
                vectordb.add_documents(chunk_batch)
                vectordb.persist()
                time.sleep(0.5)
            except Exception as e:
                print(f"[ERROR] Ошибка при загрузке чанков: {e}")

        del docs
        del chunks
        gc.collect()

    print(f"[SUCCESS] Загружено всего {total_docs} страниц, {total_chunks} чанков")
    return vectordb
