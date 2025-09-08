import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import os
import streamlit as st
from rag_assistant.main import get_or_create_vectorstore, list_documents
from rag_assistant.self_rag import run_self_rag
from rag_assistant import config
import logging

logging.basicConfig(level=logging.INFO)

# Streamlit config
st.set_page_config(
    page_title="Ассистент по СН РК",
    layout="wide"
)

st.title("Ассистент по строительным нормам РК")

def main():
    # ░█░█░█▀█░█▀▀░█▀▄░▀█▀░█▀▀░█░░░█░█
    # ░█▄█░█▀█░█▀▀░█░█░░█░░█▀▀░█░░░█░█
    # ░▀░▀░▀░▀░▀▀▀░▀▀░░▀▀▀░▀▀▀░▀▀▀░▀░▀

    with st.sidebar:
        st.header("📤 Добавить новые PDF")
        uploaded_files = st.file_uploader("Загрузите PDF-файлы", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                path = os.path.join(config.PDF_DIR, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
            st.success(f"✅ Загружено {len(uploaded_files)} новых PDF-файлов. Перезапустите страницу для обновления базы.")

    with st.spinner("🔄 Загрузка векторной базы..."):
        try:
            vectordb = get_or_create_vectorstore()
            all_documents = list_documents(vectordb)

            if not all_documents:
                st.error("❌ В векторной базе нет документов!")
                st.write(f"Проверь папку: `{config.PDF_DIR}` и логи запуска.")
                return

            st.session_state["vectordb"] = vectordb
            st.session_state["all_documents"] = all_documents

            st.success(f"✅ Загружено {len(all_documents)} документов.")
        except Exception as e:
            st.error(f"🚨 Ошибка при загрузке базы: {str(e)}")
            return

    # ░█▀▀░█▀█░█▀▀░█▀▀░█▀▀░█▀█░▀█▀░█▀▀
    # ░█░█░█░█░█░█░█░█░█▀▀░█░█░░█░░█▀▀
    # ░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀▀▀

    with st.sidebar:
        st.header("📑 Документы:")
        for doc_name in st.session_state["all_documents"]:
            st.write(f"• {doc_name}")

    selected = st.selectbox(
        "🔍 Ограничить вопрос одним документом (по названию):",
        options=["Все документы"] + st.session_state["all_documents"],
        index=0,
    )

    question = st.text_input("Введите вопрос:")

    if question.strip():
        with st.spinner("🔎 Обработка запроса (Self-RAG)..."):
            try:
                selected_doc = selected if selected != "Все документы" else None
                result = run_self_rag(
                    question=question,
                    vectordb=st.session_state["vectordb"],
                    selected_document=selected_doc,
                    model_name="gpt-4o",
                )

                st.markdown("### 🧠 Ответ:")
                st.write(result.get("answer", ""))

                # Critique block
                critique = result.get("critique", {})
                with st.expander("🧪 Критика и уверенность модели"):
                    st.write({
                        "used_retrieval": result.get("used_retrieval"),
                        "retrieve_reason": result.get("retrieve_reason"),
                        "confidence": result.get("confidence"),
                        "critique": critique,
                    })

                # Citations and snippets
                citations = result.get("citations", [])
                citation_meta = {c["doc_id"]: c for c in result.get("citation_meta", [])}
                if citations:
                    st.markdown("### 📚 Источники")
                    for c in citations:
                        did = c.get("doc_id")
                        meta = citation_meta.get(did, {})
                        source = meta.get("source", "")
                        page = meta.get("page", "")
                        st.write(f"[{did}] {source}, стр. {page}")
                        quote = c.get("quote", "")
                        if quote:
                            st.text(quote)

            except Exception as e:
                st.error(f"❌ Ошибка при обработке запроса: {str(e)}")

if __name__ == "__main__":
    main()