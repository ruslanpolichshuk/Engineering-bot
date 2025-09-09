import os
import streamlit as st
from rag_assistant.main import get_or_create_vectorstore, list_documents, create_qa_chain, extract_accurate_sources
from rag_assistant.self_rag import run_self_rag
from rag_assistant.self_rag_langgraph import run_self_rag_langgraph
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

    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.selectbox(
        "🔍 Ограничить вопрос одним документом (по названию):",
        options=["Все документы"] + st.session_state["all_documents"],
        index=0,
        )
    with col2:
        rag_mode = st.selectbox(
            "Режим RAG:",
            options=["Обычный RAG", "Self-RAG (старый)", "Self-RAG LangGraph (новый)"],
            index=2,
            help="Выберите режим работы системы"
        )

    question = st.text_input("Введите вопрос:")

    if question.strip():
        with st.spinner("🔎 Обработка запроса..."):
            try:
                if rag_mode == "Self-RAG LangGraph (новый)":
                    # Use new LangGraph-based Self-RAG
                    flow = run_self_rag_langgraph(
                        st.session_state["vectordb"],
                        question=question,
                        selected_document=selected if selected != "Все документы" else None,
                    )
                    answer = flow.get("final_answer", "")
                    accuracy = flow.get("accuracy_percentage", 0.0)
                    retrieve_decision = flow.get("retrieve_decision", "")
                    relevance_scores = flow.get("relevance_scores", {})
                    support_scores = flow.get("support_scores", {})
                    usefulness_scores = flow.get("usefulness_scores", {})

                    st.markdown("### 🧠 Ответ:")
                    st.write(answer)

                    with st.expander("🔎 Подробности Self-RAG LangGraph"):
                        st.write(f"**Решение о поиске:** {retrieve_decision}")
                        st.write(f"**Оценки релевантности:** {relevance_scores}")
                        st.write(f"**Оценки поддержки:** {support_scores}")
                        st.write(f"**Оценки полезности:** {usefulness_scores}")
                        st.write(f"**Исходная генерация:** {flow.get('generation', '')}")

                    src_docs = flow.get("source_documents", [])
                    if src_docs:
                        st.markdown("### 📚 Использованные фрагменты:")
                        accurate_sources = extract_accurate_sources(src_docs)
                        for source_info in accurate_sources:
                            st.write(f"📄 **{source_info['source']}**, стр. {source_info['page']}:")
                            st.text(source_info['content_preview'])
                            
                elif rag_mode == "Self-RAG (старый)":
                    # Use old Self-RAG implementation
                    flow = run_self_rag(
                        st.session_state["vectordb"],
                        question=question,
                        selected_document=selected if selected != "Все документы" else None,
                    )
                    answer = flow.get("final_answer", "")
                    decision = flow.get("decision", {})
                    critique = flow.get("critique", {})

                    st.markdown("### 🧠 Ответ:")
                    st.write(answer)

                    with st.expander("🔎 Подробности Self-RAG"):
                        st.write("Решение о необходимости поиска:", decision)
                        st.write("Самокритика:", critique)
                        st.write("Переформулированный запрос:", flow.get("reformulated_query", ""))

                    src_docs = flow.get("source_documents", [])
                    if src_docs:
                        st.markdown("### 📚 Использованные фрагменты:")
                        accurate_sources = extract_accurate_sources(src_docs)
                        for source_info in accurate_sources:
                            st.write(f"📄 **{source_info['source']}**, стр. {source_info['page']}:")
                            st.text(source_info['content_preview'])
                            
                else:  # Обычный RAG
                    qa_chain = create_qa_chain(
                        vectordb=st.session_state["vectordb"],
                        selected_document=selected if selected != "Все документы" else None
                    )
                    result = qa_chain({"query": question})
                    answer = result.get("result", "")

                    if "нет информации" in answer.lower():
                        st.warning("🤔 Точного ответа не найдено. Вот фрагменты, которые могут быть полезны:")
                        src_docs = result.get("source_documents", [])
                        if src_docs:
                            accurate_sources = extract_accurate_sources(src_docs)
                            for source_info in accurate_sources:
                                st.write(f"📄 **{source_info['source']}**, стр. {source_info['page']}:")
                                st.text(source_info['content_preview'])
                    else:
                        st.markdown("### 🧠 Ответ:")
                        st.write(answer)
                        
                        # Show accurate sources for regular RAG too
                        src_docs = result.get("source_documents", [])
                        if src_docs:
                            st.markdown("### 📚 Использованные фрагменты:")
                            accurate_sources = extract_accurate_sources(src_docs)
                            for source_info in accurate_sources:
                                st.write(f"📄 **{source_info['source']}**, стр. {source_info['page']}:")
                                st.text(source_info['content_preview'])

            except Exception as e:
                st.error(f"❌ Ошибка при обработке запроса: {str(e)}")

if __name__ == "__main__":
    main()