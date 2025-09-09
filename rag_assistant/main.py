import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from rag_assistant.utils import get_or_create_vectorstore_incremental as utils_get_vectorstore
from rag_assistant import config
from rag_assistant.knowledge_graph import KnowledgeGraphBuilder
from rag_assistant.kg_retriever import KnowledgeGraphRetriever
from langchain.prompts import PromptTemplate
from typing import List, Dict, Set

def extract_accurate_sources(source_documents) -> List[Dict[str, str]]:
    """
    Извлекает точные источники из документов без галлюцинаций
    """
    sources = []
    seen_sources = set()
    
    for doc in source_documents:
        source = doc.metadata.get('source', 'неизвестный источник')
        page = doc.metadata.get('page', '?')
        source_key = f"{source}_page_{page}"
        
        if source_key not in seen_sources:
            sources.append({
                'source': source,
                'page': page,
                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
            seen_sources.add(source_key)
    
    return sources

def get_or_create_vectorstore():
    vectordb = utils_get_vectorstore(
        pdf_dir=config.PDF_DIR,
        persist_dir=config.VECTORDB_DIR
    )
    print("[INFO] Векторная база успешно загружена или создана.")
    return vectordb

def get_or_create_knowledge_graph():
    """Get or create knowledge graph builder"""
    if not config.USE_KNOWLEDGE_GRAPH:
        return None
    
    try:
        kg_builder = KnowledgeGraphBuilder()
        print("[INFO] Граф знаний успешно загружен или создан.")
        return kg_builder
    except Exception as e:
        print(f"[ERROR] Ошибка при работе с графом знаний: {e}")
        return None

def get_enhanced_retriever(vectordb: Chroma, kg_builder: KnowledgeGraphBuilder = None):
    """Get enhanced retriever with knowledge graph support"""
    if kg_builder and config.USE_KNOWLEDGE_GRAPH:
        return KnowledgeGraphRetriever(vectordb, kg_builder)
    else:
        # Fallback to regular retriever
        return vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.SELF_RAG_SEARCH_K,
                "fetch_k": config.SELF_RAG_FETCH_K,
                "score_threshold": config.SELF_RAG_SCORE_THRESHOLD
            }
        )

def create_qa_chain(vectordb: Chroma, selected_document: str = None, kg_builder: KnowledgeGraphBuilder = None):
    """
    Создает RetrievalQA цепочку с улучшенным промптом и настройками,
    при необходимости ограничивает поиск одним документом
    """

    search_kwargs = {
        "k": config.SELF_RAG_SEARCH_K,
        "score_threshold": config.SELF_RAG_SCORE_THRESHOLD,
        "fetch_k": config.SELF_RAG_FETCH_K
    }

    if selected_document and selected_document != "Все документы":
        print(f"[INFO] Ограничиваем поиск документом: {selected_document}")
        # Для Chroma используем metadata-фильтр по точному совпадению источника
        search_kwargs["filter"] = {"source": selected_document}

    # Use enhanced retriever if knowledge graph is available
    retriever = get_enhanced_retriever(vectordb, kg_builder)

    llm = ChatOpenAI(
        model_name=config.SELF_RAG_MODEL,  # Now using GPT-5
        temperature=config.SELF_RAG_TEMPERATURE,
        openai_api_key=config.API_KEY
    )

    QA_PROMPT = """Ты - эксперт по строительным нормам Республики Казахстан. Ответь на вопрос, используя ТОЛЬКО предоставленные фрагменты документов. 
Даже если информация неполная, сформулируй ответ на основе того, что есть.

ВАЖНО: В разделе "Источники" указывай ТОЛЬКО реальные источники из контекста. НЕ выдумывай названия документов или номера страниц.

Контекст:
{context}

Вопрос: {question}

Ответ должен содержать: 
1. Четкий ответ на вопрос
2. Номера пунктов нормативов (если есть в контексте)
3. Различия между типами конструкций (если упоминаются)
4. Точные данные из предоставленных фрагментов
5. Использование контекста из базы знаний (если доступен)

Ответ:
Развернутый ответ:
Источники: (укажи ТОЛЬКО реальные источники из контекста выше)"""

    prompt = PromptTemplate(
        template=QA_PROMPT,
        input_variables=["context", "question"]
    )

    print("[INFO] Создаём цепочку QA с GPT-5...")
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
