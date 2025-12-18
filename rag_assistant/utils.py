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
    """
    Создает или загружает векторную базу данных.
    Оптимизировано для Railway deployment с поддержкой persistent volumes.
    """
    print(f"[INFO] get_or_create_vectorstore: persist_dir={persist_dir}, force_rebuild={force_rebuild}")
    
    # Use the latest and most accurate embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1536  # Standard dimension for compatibility
    )
    
    if not force_rebuild:
        try:
            # Пытаемся подключиться к существующей базе
            # Проверяем, что директория существует и не пуста
            if os.path.exists(persist_dir):
                try:
                    dir_contents = os.listdir(persist_dir)
                    if not dir_contents:
                        print(f"[INFO] Директория векторной базы пуста: {persist_dir}")
                    else:
                        # Пытаемся загрузить существующую базу
                        print(f"[INFO] Найдена существующая векторная база в {persist_dir} ({len(dir_contents)} файлов), проверяем...")
                        vectordb = Chroma(
                            persist_directory=persist_dir,
                            embedding_function=embeddings
                        )
                        # Check if collection has documents
                        try:
                            count = vectordb._collection.count()
                            if count > 0:
                                print(f"[INFO] ✓ Загружена существующая векторная база с {count} документами")
                                return vectordb
                            else:
                                print("[WARN] Векторная база пуста, будет пересоздана")
                        except AttributeError:
                            # Старый API ChromaDB
                            try:
                                collection_data = vectordb._collection.get()
                                count = len(collection_data.get('ids', []))
                                if count > 0:
                                    print(f"[INFO] ✓ Загружена существующая векторная база с {count} документами (старый API)")
                                    return vectordb
                                else:
                                    print("[WARN] Векторная база пуста, будет пересоздана")
                            except Exception as e:
                                print(f"[WARN] Ошибка проверки базы (старый API): {e}, будет пересоздана")
                        except Exception as e:
                            print(f"[WARN] Ошибка проверки базы: {e}, будет пересоздана")
                except OSError as e:
                    print(f"[WARN] Ошибка доступа к директории {persist_dir}: {e}, будет пересоздана")
            else:
                print(f"[INFO] Директория векторной базы не существует: {persist_dir}, будет создана")
        except Exception as e:
            print(f"[WARN] Ошибка загрузки базы: {e}, будет пересоздана")

    # Если нужно пересоздать или база повреждена
    print("[INFO] Создание новой векторной базы...")
    for attempt in range(3):
        try:
            if force_rebuild:
                if os.path.exists(persist_dir):
                    print(f"[INFO] Удаление старой базы: {persist_dir}")
                    shutil.rmtree(persist_dir, ignore_errors=True)
           
            # Ensure directory exists (important for Railway volumes)
            os.makedirs(persist_dir, exist_ok=True)
            print(f"[INFO] Создана директория: {persist_dir}")
            
            # Проверяем наличие PDF файлов
            if not os.path.exists(pdf_dir):
                print(f"[WARN] Директория PDF не найдена: {pdf_dir}, создаем...")
                os.makedirs(pdf_dir, exist_ok=True)
            
            pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')] if os.path.exists(pdf_dir) else []
            if not pdf_files:
                print(f"[WARN] PDF файлы не найдены в {pdf_dir}")
                # Return empty vectorstore if no PDFs
                vectordb = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=embeddings
                )
                return vectordb
            
            # Создаем новую базу
            print(f"[INFO] Обработка {len(pdf_files)} PDF файлов...")
            docs = parse_pdfs(pdf_dir)
            if not docs:
                print("[ERROR] Не удалось загрузить документы из PDF")
                # Return empty vectorstore
                vectordb = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=embeddings
                )
                return vectordb
            
            # Improved chunking strategy for better retrieval accuracy
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Increased for better context
                chunk_overlap=200,  # Increased overlap for continuity
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]  # Better separation for technical documents
            )
            chunks = splitter.split_documents(docs)
            print(f"[INFO] Создано {len(chunks)} чанков из {len(docs)} документов")
            
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_dir
            )
            print(f"[INFO] ✓ Векторная база успешно создана в {persist_dir}")
            return vectordb
            
        except PermissionError as e:
            if attempt == 2:
                print(f"[ERROR] Ошибка прав доступа после 3 попыток: {e}")
                raise
            print(f"[WARN] Ошибка прав доступа, попытка {attempt + 1}/3: {e}")
            time.sleep(1)  # Ждем 1 сек перед повторной попыткой
        except Exception as e:
            if attempt == 2:
                print(f"[ERROR] Критическая ошибка создания базы: {e}")
                raise
            print(f"[WARN] Ошибка создания базы, попытка {attempt + 1}/3: {e}")
            time.sleep(1)
