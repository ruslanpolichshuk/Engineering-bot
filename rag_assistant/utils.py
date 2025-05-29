import os
import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
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


import time
import shutil
from retrying import retry

@retry(stop_max_attempt_number=3, wait_fixed=1000)
def get_or_create_vectorstore(pdf_dir, persist_dir, force_rebuild=False):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    if not force_rebuild:
        try:
            # Пытаемся подключиться к существующей базе
            vectordb = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
            if vectordb._collection.count() > 0:
                return vectordb
        except Exception as e:
            print(f"[WARN] Ошибка загрузки базы: {e}")

    # Если нужно пересоздать или база повреждена
    for attempt in range(3):
        try:
            if force_rebuild:
                if os.path.exists(persist_dir):
                    shutil.rmtree(persist_dir, ignore_errors=True)
           
            os.makedirs(persist_dir, exist_ok=True)
            
            # Создаем новую базу
            docs = parse_pdfs(pdf_dir)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_dir
            )
            return vectordb
            
        except PermissionError as e:
            if attempt == 2:
                raise
            time.sleep(1)  # Ждем 1 сек перед повторной попыткой
