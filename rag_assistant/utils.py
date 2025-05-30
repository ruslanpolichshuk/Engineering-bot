import os
import time
import shutil
import pdfplumber
from retrying import retry
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
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
            print(f"[LOAD] Открываем: {fname}")
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text or len(text.strip()) < 20:
                        print(f"[WARN] Пустая/короткая страница {i+1} в {fname}")
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
def get_or_create_vectorstore(pdf_dir, persist_dir, force_rebuild=False):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        batch_size=16,           # ✅ batch вместо одного запроса
        request_timeout=60       # ✅ чуть выше таймаут на всякий случай
    )

    if not force_rebuild:
        try:
            vectordb = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
            count = vectordb._collection.count()
            print(f"[INFO] Найдено {count} векторов в существующей базе")
            if count > 0:
                return vectordb
        except Exception as e:
            print(f"[WARN] Ошибка подключения к существующей базе: {e}")

    # Пересоздание базы
    for attempt in range(3):
        try:
            if force_rebuild:
                print(f"[INFO] Пересоздание базы. Удаляем {persist_dir}")
                if os.path.exists(persist_dir):
                    shutil.rmtree(persist_dir, ignore_errors=True)

            os.makedirs(persist_dir, exist_ok=True)

            print("[INFO] Загружаем документы...")
            docs = parse_pdfs(pdf_dir)
            if not docs:
                raise ValueError("❌ Ни одного валидного PDF-документа не загружено.")

            print("[INFO] Разбиваем на чанки...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            print(f"[INFO] Всего чанков для эмбеддинга: {len(chunks)}")

            print("[INFO] Начинаем построение векторной базы...")
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_dir
            )
            print("[SUCCESS] Векторная база успешно создана и сохранена.")
            return vectordb

        except PermissionError as e:
            print(f"[RETRY] PermissionError: {e}")
            if attempt == 2:
                raise
            time.sleep(1)

        except Exception as e:
            print(f"[ERROR] Ошибка при создании базы: {e}")
            if attempt == 2:
                raise
            time.sleep(1)
