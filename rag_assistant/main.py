import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from rag_assistant.utils import get_or_create_vectorstore as utils_get_vectorstore
from rag_assistant import config
from langchain.prompts import PromptTemplate

def get_or_create_vectorstore(force_rebuild: bool = False):
    vectordb = utils_get_vectorstore(
        pdf_dir=config.PDF_DIR,
        persist_dir=config.VECTORDB_DIR,
        force_rebuild=force_rebuild
    )
    print("[INFO] Векторная база успешно загружена или создана.")
    return vectordb

def create_qa_chain(vectordb: Chroma, selected_document: str = None):
    """
    Создает RetrievalQA цепочку с улучшенным промптом и настройками,
    при необходимости ограничивает поиск одним документом
    """

    search_kwargs = {
        "k": 10,
        "score_threshold": 0.4,
        "fetch_k": 30
    }

    if selected_document and selected_document != "Все документы":
        print(f"[INFO] Ограничиваем поиск документом: {selected_document}")
        search_kwargs["filter"] = lambda doc: doc.metadata.get("source") == selected_document

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        openai_api_key=config.API_KEY
    )

    QA_PROMPT = """Ты - эксперт по строительным нормам. Ответь на вопрос, используя ТОЛЬКО предоставленные фрагменты документов. 
Даже если информация неполная, сформулируй ответ на основе того, что есть.

Контекст:
{context}

Вопрос: {question}

Ответ должен содержать: 
1. Четкий ответ на вопрос
2. Номера пунктов нормативов (если есть)
3. Различия между типами конструкций (если упоминаются)
4. Имя источника и точные данные из документов. Имя источника - название документа (СН РК Х.ХХ-ХХ-ХХХХ) 

Ответ:
Развернутый ответ:
Источники:"""

    prompt = PromptTemplate(
        template=QA_PROMPT,
        input_variables=["context", "question"]
    )

    print("[INFO] Создаём цепочку QA...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("[INFO] QA-цепочка успешно создана.")
    return qa_chain

def list_documents(vectordb) -> list[str]:
    try:
        collection = vectordb.get()
        metadatas = collection.get("metadatas", [])
        sources = list({m['source'] for m in metadatas if isinstance(m, dict) and 'source' in m})
        print(f"[INFO] Получено {len(sources)} уникальных документов.")
        return sorted(sources)
    except Exception as e:
        print(f"[ERROR] list_documents: {str(e)}")
        return []
