import os
import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pdfminer.pdfparser import PDFSyntaxError
from rag_assistant import config

# Импорт блокировки файлов (разный для разных ОС)
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

if os.name == 'nt':
    try:
        import msvcrt
        HAS_MSVCRT = True
    except ImportError:
        HAS_MSVCRT = False
else:
    HAS_MSVCRT = False


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

def _check_vectorstore_exists(persist_dir, embeddings):
    """Проверяет существование и валидность векторной базы"""
    if not os.path.exists(persist_dir):
        return False, None
    
    dir_contents = os.listdir(persist_dir)
    if not dir_contents:
        return False, None
    
    # Проверяем наличие ключевых файлов ChromaDB
    # ChromaDB может использовать разные структуры:
    # 1. Старая: chroma.sqlite3, chroma.sqlite3-wal
    # 2. Новая: поддиректории с коллекциями
    has_sqlite = any('sqlite' in f.lower() for f in dir_contents)
    has_subdirs = any(os.path.isdir(os.path.join(persist_dir, item)) for item in dir_contents)
    
    if not has_sqlite and not has_subdirs:
        return False, None
    
    try:
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        
        # Проверяем количество документов разными способами
        count = 0
        try:
            # Новый API ChromaDB
            if hasattr(vectordb._collection, 'count'):
                count = vectordb._collection.count()
            elif hasattr(vectordb._collection, 'get'):
                # Старый API - получаем все документы
                collection_data = vectordb._collection.get()
                ids = collection_data.get('ids', [])
                if ids:
                    count = len(ids)
            else:
                # Пытаемся через peek
                try:
                    peek_data = vectordb._collection.peek(limit=1)
                    if peek_data and peek_data.get('ids'):
                        # Если есть хотя бы один документ, пробуем получить все
                        all_data = vectordb._collection.get()
                        count = len(all_data.get('ids', []))
                except:
                    pass
        except Exception as e:
            print(f"[DEBUG] Ошибка при подсчете документов: {e}")
            # Пробуем альтернативный способ
            try:
                # Пытаемся получить хотя бы один документ
                results = vectordb.similarity_search("test", k=1)
                if results:
                    count = 1  # Если можем получить результаты, значит база не пуста
                    # Но лучше получить точное количество
                    try:
                        all_data = vectordb._collection.get()
                        count = len(all_data.get('ids', []))
                    except:
                        pass
            except:
                pass
        
        if count > 0:
            print(f"[INFO] Найдено {count} документов в базе")
            return True, vectordb
        else:
            print(f"[WARN] База существует, но пуста (count={count})")
            return False, None
            
    except Exception as e:
        print(f"[WARN] Ошибка проверки базы: {e}")
        import traceback
        traceback.print_exc()
        return False, None

@retry(stop_max_attempt_number=3, wait_fixed=1000)
def get_or_create_vectorstore(pdf_dir, persist_dir, force_rebuild=False):
    """
    Создает или загружает векторную базу данных.
    Оптимизировано для Railway deployment с поддержкой persistent volumes.
    Использует файловую блокировку для предотвращения параллельного создания.
    """
    print(f"[INFO] get_or_create_vectorstore: persist_dir={persist_dir}, force_rebuild={force_rebuild}")
    
    # Use the latest and most accurate embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1536  # Standard dimension for compatibility
    )
    
    # Файл блокировки для предотвращения параллельного создания
    lock_file_path = os.path.join(persist_dir, ".vectordb.lock")
    lock_file = None
    
    try:
        # Пытаемся загрузить существующую базу
        if not force_rebuild:
            exists, vectordb = _check_vectorstore_exists(persist_dir, embeddings)
            if exists:
                print(f"[INFO] ✓ Загружена существующая векторная база")
                return vectordb
            else:
                print(f"[INFO] Векторная база не найдена или пуста, будет создана")
        
        # Проверяем, не создается ли база другим процессом
        if os.path.exists(lock_file_path):
            # Проверяем время модификации файла блокировки
            lock_age = time.time() - os.path.getmtime(lock_file_path)
            if lock_age < 3600:  # Если блокировка свежая (меньше часа), ждем
                print(f"[INFO] Другой процесс создает базу (блокировка {lock_age:.0f} сек назад), ожидание...")
                max_wait = 300  # Максимум 5 минут ожидания
                waited = 0
                while waited < max_wait and os.path.exists(lock_file_path):
                    time.sleep(5)
                    waited += 5
                    # Проверяем, не создалась ли база за это время
                    exists, vectordb = _check_vectorstore_exists(persist_dir, embeddings)
                    if exists:
                        print(f"[INFO] ✓ База создана другим процессом, загружаем...")
                        return vectordb
                
                # Если прошло много времени, возможно процесс завис
                if lock_age > 3600:
                    print(f"[WARN] Блокировка старая ({lock_age:.0f} сек), возможно процесс завис, продолжаем...")
                    try:
                        os.remove(lock_file_path)
                    except:
                        pass
            else:
                # Старая блокировка, удаляем
                try:
                    os.remove(lock_file_path)
                except:
                    pass
        
        # Создаем блокировку
        os.makedirs(persist_dir, exist_ok=True)
        lock_file = open(lock_file_path, 'w')
        lock_acquired = False
        
        try:
            if HAS_FCNTL:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True
            elif HAS_MSVCRT:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                lock_acquired = True
            else:
                # Если нет поддержки блокировки, просто создаем файл
                # Это не идеально, но лучше чем ничего
                lock_acquired = True
        except (IOError, OSError) as e:
            # Блокировка уже установлена другим процессом
            lock_file.close()
            lock_file = None
            print(f"[INFO] База создается другим процессом (блокировка занята), ожидание...")
            # Ждем и проверяем снова
            time.sleep(10)
            exists, vectordb = _check_vectorstore_exists(persist_dir, embeddings)
            if exists:
                return vectordb
            raise Exception("Не удалось получить блокировку для создания базы")
        
        if lock_acquired:
            lock_file.write(str(time.time()))
            lock_file.flush()
            print(f"[INFO] Блокировка установлена для создания базы")

        
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
                print(f"[INFO] Разбиваем документы на чанки...")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,  # Increased for better context
                    chunk_overlap=200,  # Increased overlap for continuity
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]  # Better separation for technical documents
                )
                chunks = splitter.split_documents(docs)
                print(f"[INFO] Создано {len(chunks)} чанков из {len(docs)} документов")
                
                # Check disk space before creating embeddings
                if os.path.exists(persist_dir):
                    total, used, free = shutil.disk_usage(persist_dir)
                    print(f"[INFO] Дисковое пространство: использовано {used // (1024**2)} MB, свободно {free // (1024**2)} MB")
                
                print(f"[INFO] Генерация эмбеддингов через OpenAI API (это может занять несколько минут для {len(chunks)} чанков)...")
                start_time = time.time()
                
                # Создаем базу (ChromaDB автоматически сохраняет при указании persist_directory)
                vectordb = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=persist_dir
                )
                
                # Даем время на сохранение
                time.sleep(2)
                
                elapsed_time = time.time() - start_time
                print(f"[INFO] ✓ Векторная база успешно создана в {persist_dir} (заняло {elapsed_time:.1f} секунд)")
                
                # Verify persistence and check disk usage
                if os.path.exists(persist_dir):
                    dir_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                 for dirpath, dirnames, filenames in os.walk(persist_dir)
                                 for filename in filenames)
                    print(f"[INFO] Размер векторной базы на диске: {dir_size // (1024**2)} MB")
                    file_count = sum([len(files) for r, d, files in os.walk(persist_dir)])
                    print(f"[INFO] Количество файлов в базе: {file_count}")
                else:
                    print(f"[WARN] Директория {persist_dir} не существует после создания базы!")
                
                # Проверяем, что база действительно создана
                exists, verified_vectordb = _check_vectorstore_exists(persist_dir, embeddings)
                if exists:
                    print(f"[INFO] ✓ База проверена и готова к использованию")
                    return verified_vectordb
                else:
                    raise Exception("База не прошла проверку после создания")
                
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
    finally:
        # Освобождаем блокировку
        if lock_file:
            try:
                lock_file.close()
            except:
                pass
        if os.path.exists(lock_file_path):
            try:
                os.remove(lock_file_path)
            except:
                pass
