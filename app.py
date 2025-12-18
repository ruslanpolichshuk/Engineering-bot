import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from rag_assistant.main import get_or_create_vectorstore, list_documents, create_qa_chain
from rag_assistant import config
import logging
# Отключаем DEBUG логи для всех библиотек, кроме наших важных сообщений
logging.basicConfig(level=logging.WARNING)
# Отключаем особенно шумные логи от pdfminer
logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('pdfplumber').setLevel(logging.WARNING)

# Должен быть ПЕРВЫМ вызовом в скрипте
st.set_page_config(
    page_title="Ассистент по строительным PDF СН РК",
    layout="wide"
)

st.title("Ассистент по строительным нормам РК")

@st.cache_resource(ttl=3600)  # Кэшируем на 1 час
def load_vectorstore():
    """Загружает векторную базу с кэшированием"""
    return get_or_create_vectorstore()

def main():
    # Загрузка или создание векторной базы
    if 'vectordb' not in st.session_state or 'all_documents' not in st.session_state:
        with st.spinner("🔄 Загрузка векторной базы..."):
            try:
                vectordb = load_vectorstore()
                all_documents = list_documents(vectordb)
                
                if not all_documents:
                    st.error("⚠️ В векторной базе нет документов! Проверьте:")
                    st.write(f"- Папку с PDF: {config.PDF_DIR}")
                    st.write("- Логи загрузки (должны быть в терминале)")
                    st.write("- Возможно, база еще создается (это может занять несколько минут)")
                    return
                
                st.session_state['vectordb'] = vectordb
                st.session_state['all_documents'] = all_documents
                
            except Exception as e:
                st.error(f"Ошибка загрузки: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Боковая панель: список документов
    with st.sidebar:
        st.header("📑 Все документы:")
        st.write(f"Найдено: {len(st.session_state['all_documents'])} документов.")
        for name in st.session_state['all_documents']:
            st.write(f"• {name}")

    # Основной интерфейс
    selected = st.selectbox(
        "🔍 Ограничить вопрос одним документом (по названию):",
        options=["Все документы"] + st.session_state['all_documents'],
        index=0,
    )

    question = st.text_input("Введите вопрос:")

    if question:
        with st.spinner("🔎 Обработка запроса..."):
            try:
                # Создаем цепочку с правильными параметрами
                qa_chain = create_qa_chain(st.session_state['vectordb'])
                
                # Отправляем запрос
                result = qa_chain({"query": question})
                
                # Проверяем ответ
                answer = result.get("result", "")
                if "нет информации" in answer.lower():
                    st.warning("В документах не найдено точного ответа, но есть похожая информация:")
                    # Выводим все найденные документы
                    for doc in result.get("source_documents", []):
                        st.write(f"📄 {doc.metadata['source']}, стр. {doc.metadata['page']}:")
                        st.text(doc.page_content[:500] + "...")
                else:
                    st.markdown("### 🧠 Ответ:")
                    st.write(answer)
                    
                    # Вывод источников остается без изменений
                    
            except Exception as e:
                st.error(f"Ошибка обработки запроса: {str(e)}")

if __name__ == "__main__":
    main()
