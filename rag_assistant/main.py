import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from rag_assistant.utils import get_or_create_vectorstore as utils_get_vectorstore
from rag_assistant import config
from langchain.prompts import PromptTemplate

def get_or_create_vectorstore(force_rebuild: bool = False):
    return utils_get_vectorstore(
        pdf_dir=config.PDF_DIR,
        persist_dir=config.VECTORDB_DIR,
        force_rebuild=force_rebuild
    )

def create_qa_chain(vectordb: Chroma):
    """
    Создает улучшенную цепочку RetrievalQA с правильной обработкой контекста
    """

    retriever = vectordb.as_retriever(
        search_type="mmr",  # Используем Maximal Marginal Relevance
        search_kwargs={
            "k": 10,  # Увеличиваем количество документов
            "score_threshold": 0.4,  # Более мягкий порог релевантности
            "fetch_k": 30  # Больший пул для MMR
        }
    )
    
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,  # Для более точных ответов
        openai_api_key=config.API_KEY
    )
    
    # Улучшенный промпт для ответов
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
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o", temperature=0),
        retriever=vectordb.as_retriever(search_kwargs={"k": 15}),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


# В main.py обновите list_documents:
def list_documents(vectordb) -> list[str]:
    try:
        # Новый способ получения источников
        collection = vectordb.get()
        sources = list({m['source'] for m in collection['metadatas'] if 'source' in m})
        return sorted(sources)
    except Exception as e:
        print(f"[ERROR] list_documents: {str(e)}")
        return []