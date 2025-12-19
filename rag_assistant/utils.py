import os
import logging
import pdfplumber
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pdfminer.pdfparser import PDFSyntaxError
from rag_assistant import config

# Отключаем telemetry ChromaDB для избежания ошибок
logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)

# Отключаем telemetry через переменную окружения (если поддерживается)
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')

# Опциональный импорт OCR библиотек
try:
    from pdf2image import convert_from_path
    import pytesseract
    
    # Проверяем, доступен ли Tesseract
    try:
        pytesseract.get_tesseract_version()
        OCR_AVAILABLE = True
        print("[INFO] OCR доступен (Tesseract установлен)")
    except Exception:
        OCR_AVAILABLE = False
        print("[WARN] OCR библиотеки установлены, но Tesseract не найден в системе.")
        print("[WARN] Для работы OCR установите Tesseract OCR:")
        print("[WARN]   Linux: sudo apt-get install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng")
        print("[WARN]   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("[WARN]   Mac: brew install tesseract tesseract-lang")
except ImportError:
    OCR_AVAILABLE = False
    print("[INFO] OCR библиотеки не установлены. Сканированные страницы будут пропущены.")

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


def _extract_text_with_ocr(pdf_path: str, page_num: int) -> str:
    """Извлекает текст из страницы PDF с помощью OCR"""
    if not OCR_AVAILABLE:
        return None
    
    try:
        # Конвертируем конкретную страницу в изображение
        images = convert_from_path(
            pdf_path,
            first_page=page_num,
            last_page=page_num,
            dpi=300,  # Высокое разрешение для лучшего качества OCR
            thread_count=1
        )
        
        if not images:
            return None
        
        # Применяем OCR к изображению
        text = pytesseract.image_to_string(
            images[0],
            lang='rus+eng',  # Поддержка русского и английского
            config='--psm 6'  # Предполагаем единый блок текста
        )
        
        return text.strip() if text else None
        
    except Exception as e:
        print(f"[WARN] OCR ошибка на странице {page_num}: {e}")
        return None


def parse_pdfs(pdf_dir: str) -> list[Document]:
    docs: list[Document] = []
    total_files = 0
    total_pages = 0
    total_empty = 0
    total_ocr = 0
    
    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith('.pdf'):
            continue
        total_files += 1
        path = os.path.join(pdf_dir, fname)
        try:
            print(f"[LOAD] Загружаем PDF: {fname}")
            empty_pages = []
            ocr_pages = []
            file_docs = 0
            
            with pdfplumber.open(path) as pdf:
                num_pages = len(pdf.pages)
                total_pages += num_pages
                
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    
                    # Если текст пустой или слишком короткий, пробуем OCR
                    if not text or not text.strip() or len(text.strip()) < 10:
                        empty_pages.append(i + 1)
                        
                        # Пробуем OCR для сканированных страниц
                        if OCR_AVAILABLE:
                            ocr_text = _extract_text_with_ocr(path, i + 1)
                            if ocr_text and len(ocr_text.strip()) >= 10:
                                # OCR успешно распознал текст
                                meta = {'source': fname, 'page': i + 1, 'ocr': True}
                                docs.append(Document(page_content=ocr_text, metadata=meta))
                                file_docs += 1
                                total_ocr += 1
                                ocr_pages.append(i + 1)
                                continue
                        
                        # OCR не помог или недоступен
                        total_empty += 1
                        continue
                    
                    # Обычный текст из PDF
                    meta = {'source': fname, 'page': i + 1, 'ocr': False}
                    docs.append(Document(page_content=text, metadata=meta))
                    file_docs += 1
                
                # Выводим статистику по файлу
                stats_parts = []
                if empty_pages and not ocr_pages:
                    # Только пустые страницы, без OCR
                    if len(empty_pages) <= 5:
                        stats_parts.append(f"пустые: {', '.join(map(str, empty_pages))}")
                    else:
                        stats_parts.append(f"пустых: {len(empty_pages)}")
                elif ocr_pages:
                    # Есть OCR страницы
                    if len(ocr_pages) <= 5:
                        stats_parts.append(f"OCR: {', '.join(map(str, ocr_pages))}")
                    else:
                        stats_parts.append(f"OCR: {len(ocr_pages)}")
                    if len(empty_pages) > len(ocr_pages):
                        remaining = len(empty_pages) - len(ocr_pages)
                        stats_parts.append(f"пустых: {remaining}")
                
                if stats_parts:
                    print(f"[INFO] Загружено {file_docs}/{num_pages} страниц ({', '.join(stats_parts)})")
                else:
                    print(f"[INFO] Загружено {file_docs} страниц")
                    
        except PDFSyntaxError:
            print(f"[ERROR] Повреждён PDF: {fname}")
        except Exception as e:
            print(f"[ERROR] Ошибка с {fname}: {e}")
    
    # Итоговая статистика
    print(f"[RESULT] Обработано {total_files} PDF-файлов:")
    print(f"  - Всего страниц: {total_pages}")
    print(f"  - Загружено с текстом: {len(docs)}")
    if total_ocr > 0:
        print(f"  - Распознано через OCR: {total_ocr}")
    if total_pages > 0:
        empty_percent = total_empty * 100 / total_pages
        print(f"  - Пустых/не распознанных: {total_empty} ({empty_percent:.1f}%)")
    else:
        print(f"  - Пустых/не распознанных: {total_empty}")
    
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
    
    # Выбор модели эмбеддингов: баланс скорости и точности
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    
    # text-embedding-3-small: быстрее, но менее точная
    # text-embedding-3-large: медленнее, но более точная
    # Можно использовать small для скорости или large для точности
    
    print(f"[INFO] Используется модель эмбеддингов: {embedding_model} (размерность: {embedding_dimensions})")
    
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        dimensions=embedding_dimensions,
        # Оптимизация для батчей
        chunk_size=100,  # Размер батча для эмбеддингов (OpenAI рекомендует 100-1000)
        max_retries=3,
        request_timeout=60
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
                
                # Оптимизированная стратегия разбиения на чанки для баланса скорости и точности
                print(f"[INFO] Разбиваем документы на чанки...")
                
                # Настройки из конфига или умолчания
                chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))  # Оптимальный размер для технических документов
                chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "300"))  # Больше overlap для лучшего контекста
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,  # 25% overlap для лучшего контекста
                    length_function=len,
                    separators=[
                        "\n\n\n",  # Разделы документов
                        "\n\n",    # Параграфы
                        "\n",      # Строки
                        ". ",      # Предложения
                        " ",       # Слова
                        ""         # Символы
                    ],
                    keep_separator=True  # Сохраняем разделители для контекста
                )
                chunks = splitter.split_documents(docs)
                
                # Улучшаем метаданные для лучшей точности поиска
                for i, chunk in enumerate(chunks):
                    # Добавляем информацию о позиции в документе
                    if 'page' in chunk.metadata:
                        chunk.metadata['chunk_index'] = i
                        # Добавляем префикс с номером документа для лучшей идентификации
                        source = chunk.metadata.get('source', '')
                        if source:
                            # Извлекаем номер документа из названия (СН РК X.XX-XX-XXXX)
                            import re
                            doc_number = re.search(r'СН РК [\d.]+-[\d.]+-[\d]+', source)
                            if doc_number:
                                chunk.metadata['doc_number'] = doc_number.group()
                
                print(f"[INFO] Создано {len(chunks)} чанков из {len(docs)} документов")
                print(f"[INFO] Средний размер чанка: {sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0} символов")
                
                # Check disk space before creating embeddings
                if os.path.exists(persist_dir):
                    total, used, free = shutil.disk_usage(persist_dir)
                    print(f"[INFO] Дисковое пространство: использовано {used // (1024**2)} MB, свободно {free // (1024**2)} MB")
                
                print(f"[INFO] Генерация эмбеддингов через OpenAI API (это может занять несколько минут для {len(chunks)} чанков)...")
                print(f"[INFO] Используем батчинг для избежания лимита токенов (макс. 300000 токенов на запрос)")
                start_time = time.time()
                
                # Создаем пустую базу
                vectordb = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=embeddings
                )
                
                # Оптимизированный батчинг с учетом модели и размера чанков
                # OpenAI API имеет лимит 300000 токенов на запрос
                # text-embedding-3-large: ~1 токен на 4 символа
                # Средний размер чанка ~1200 символов = ~300 токенов
                # Безопасный батч: 300000 / 300 = ~1000 чанков, но используем 800 для запаса
                
                avg_chunk_size = sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 1200
                tokens_per_chunk = avg_chunk_size // 4  # Примерно 1 токен на 4 символа
                safe_batch_size = min(800, (250000 // tokens_per_chunk) if tokens_per_chunk > 0 else 500)
                
                batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", str(safe_batch_size)))
                use_parallel = os.getenv("USE_PARALLEL_EMBEDDINGS", "false").lower() == "true"
                max_workers = int(os.getenv("MAX_WORKERS", "3"))  # Параллельные запросы к API
                
                print(f"[INFO] Размер батча: {batch_size} чанков (средний размер чанка: {avg_chunk_size} символов, ~{tokens_per_chunk} токенов)")
                if use_parallel:
                    print(f"[INFO] Параллельная обработка: {max_workers} потоков")
                
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                
                # Последовательная обработка батчей
                # ВАЖНО: ChromaDB не потокобезопасен для параллельной записи
                # Используем последовательную обработку с оптимизированными батчами
                for batch_idx in range(total_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, len(chunks))
                        batch = chunks[start_idx:end_idx]
                        
                        print(f"[INFO] Обработка батча {batch_idx + 1}/{total_batches} ({len(batch)} чанков, {start_idx+1}-{end_idx})...")
                        
                        try:
                            # Добавляем батч в базу
                            vectordb.add_documents(batch)
                            
                            # Периодически сохраняем прогресс (ChromaDB автоматически сохраняет, но можно явно вызвать)
                            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                                try:
                                    if hasattr(vectordb, 'persist'):
                                        vectordb.persist()
                                except:
                                    pass  # persist может не существовать в новых версиях
                                print(f"[INFO] Прогресс: {end_idx}/{len(chunks)} чанков обработано ({(end_idx*100)//len(chunks)}%)")
                            
                            # Небольшая задержка между батчами для избежания rate limits
                            if batch_idx < total_batches - 1:
                                time.sleep(0.3)  # Уменьшена задержка для ускорения
                                
                        except Exception as batch_error:
                            error_msg = str(batch_error)
                            
                            # Обработка лимита токенов
                            if "max_tokens_per_request" in error_msg or "300000" in error_msg:
                                print(f"[WARN] Лимит токенов достигнут для батча {batch_idx + 1}, уменьшаем размер...")
                                smaller_batch_size = max(batch_size // 2, 10)
                                
                                # Разбиваем текущий батч на меньшие части
                                for sub_batch_start in range(start_idx, end_idx, smaller_batch_size):
                                    sub_batch_end = min(sub_batch_start + smaller_batch_size, end_idx)
                                    sub_batch = chunks[sub_batch_start:sub_batch_end]
                                    retry_count = 0
                                    while retry_count < 3:
                                        try:
                                            vectordb.add_documents(sub_batch)
                                            try:
                                                if hasattr(vectordb, 'persist'):
                                                    vectordb.persist()
                                            except:
                                                pass
                                            break
                                        except Exception as sub_error:
                                            retry_count += 1
                                            if retry_count >= 3:
                                                print(f"[ERROR] Не удалось обработать под-батч {sub_batch_start}-{sub_batch_end} после 3 попыток: {sub_error}")
                                                # Пропускаем проблемный батч
                                                break
                                            time.sleep(2 ** retry_count)  # Экспоненциальная задержка
                                
                                # Обновляем размер батча для следующих итераций
                                batch_size = smaller_batch_size
                            
                            # Обработка rate limits
                            elif "rate_limit" in error_msg.lower() or "429" in error_msg:
                                print(f"[WARN] Rate limit достигнут, ожидание 60 секунд...")
                                time.sleep(60)
                                # Повторяем текущий батч
                                batch_idx -= 1
                                continue
                            
                            # Другие ошибки
                            else:
                                print(f"[ERROR] Ошибка при обработке батча {batch_idx + 1}: {batch_error}")
                                # Пробуем повторить с меньшим батчем
                                if batch_size > 10:
                                    print(f"[WARN] Пробуем повторить с меньшим батчем...")
                                    batch_size = max(batch_size // 2, 10)
                                    batch_idx -= 1
                                    continue
                                else:
                                    print(f"[ERROR] Критическая ошибка, пропускаем батч {batch_idx + 1}")
                                    # Пропускаем проблемный батч и продолжаем
                                    continue
                
                # Финальное сохранение (ChromaDB автоматически сохраняет при persist_directory)
                try:
                    if hasattr(vectordb, 'persist'):
                        vectordb.persist()
                except:
                    pass
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
