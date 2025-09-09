from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from rag_assistant import config


def _build_retriever(vectordb, selected_document: Optional[str] = None):
    search_kwargs: Dict = {
        "k": getattr(config, "SELF_RAG_SEARCH_K", 10),
        "fetch_k": getattr(config, "SELF_RAG_FETCH_K", 30),
        "score_threshold": getattr(config, "SELF_RAG_SCORE_THRESHOLD", 0.4),
    }

    if selected_document and selected_document != "Все документы":
        search_kwargs["filter"] = {"source": selected_document}

    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )


def _format_context(documents: List[Document], max_chars: int = 12000) -> str:
    parts: List[str] = []
    current_len = 0
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "неизвестный источник")
        page = doc.metadata.get("page", "?")
        header = f"[ФРАГМЕНТ {i}] Источник: {source}, стр. {page}"
        chunk = f"{header}\n{doc.page_content.strip()}"
        if current_len + len(chunk) > max_chars:
            break
        parts.append(chunk)
        current_len += len(chunk)
    return "\n\n---\n\n".join(parts)


def _parse_yes_no(text: str) -> str:
    value = text.strip().lower()
    if "yes" in value or "да" in value:
        return "YES"
    if "no" in value or "нет" in value:
        return "NO"
    return "UNKNOWN"


def _parse_decision(text: str) -> Dict[str, str]:
    decision, reason = "UNKNOWN", text.strip()
    lowered = text.lower()
    if "decision:" in lowered:
        try:
            lines = [l for l in text.splitlines() if l.strip()]
            for line in lines:
                if line.lower().startswith("decision:") or line.lower().startswith("решение:"):
                    decision = _parse_yes_no(line.split(":", 1)[1])
                if line.lower().startswith("reason:") or line.lower().startswith("обоснование:"):
                    reason = line.split(":", 1)[1].strip()
                    break
        except Exception:
            decision = "UNKNOWN"
    else:
        decision = _parse_yes_no(text)
    return {"decision": decision, "reason": reason}


def _parse_critique(text: str) -> Dict[str, str]:
    result = {
        "evidence_sufficiency": "UNKNOWN",
        "need_more_retrieval": "UNKNOWN",
        "hallucination_risk": "UNKNOWN",
        "suggested_terms": "",
        "notes": text.strip(),
    }
    try:
        lines = [l for l in text.splitlines() if l.strip()]
        for line in lines:
            lower = line.lower()
            if lower.startswith("evidence_sufficiency:") or lower.startswith("достаточность_доказательств:"):
                result["evidence_sufficiency"] = _parse_yes_no(line.split(":", 1)[1])
            elif lower.startswith("need_more_retrieval:") or lower.startswith("нужен_доп_поиск:"):
                result["need_more_retrieval"] = _parse_yes_no(line.split(":", 1)[1])
            elif lower.startswith("hallucination_risk:") or lower.startswith("риск_галлюцинаций:"):
                result["hallucination_risk"] = line.split(":", 1)[1].strip().upper()
            elif lower.startswith("suggested_terms:") or lower.startswith("доп_термины:"):
                result["suggested_terms"] = line.split(":", 1)[1].strip()
            elif lower.startswith("notes:") or lower.startswith("заметки:"):
                result["notes"] = line.split(":", 1)[1].strip()
    except Exception:
        pass
    return result


def _uniq_documents(documents: List[Document], limit: int) -> List[Document]:
    seen: set = set()
    result: List[Document] = []
    for doc in documents:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        result.append(doc)
        if len(result) >= limit:
            break
    return result


def run_self_rag(vectordb, question: str, selected_document: Optional[str] = None) -> Dict:
    llm = ChatOpenAI(
        model_name=getattr(config, "SELF_RAG_MODEL", "gpt-4o-mini"),
        temperature=getattr(config, "SELF_RAG_TEMPERATURE", 0),
        openai_api_key=config.API_KEY,
    )

    max_rounds: int = getattr(config, "SELF_RAG_MAX_ROUNDS", 2)
    max_chunks: int = getattr(config, "SELF_RAG_MAX_CONTEXT_CHUNKS", 10)

    decide_prompt = PromptTemplate(
        template=(
            "Ты — эксперт-классификатор по строительным нормам и техническим документам.\n"
            "Определи, требует ли вопрос обращения к нормативным документам для получения точного технического ответа.\n\n"
            "Ответь YES если вопрос касается:\n"
            "- Конкретных технических требований, норм, стандартов\n"
            "- Параметров, размеров, характеристик конструкций\n"
            "- Методов расчета, формул, коэффициентов\n"
            "- Классификации, категорий, типов\n"
            "- Процедур, последовательности действий\n\n"
            "Ответь NO если вопрос:\n"
            "- Общий или теоретический\n"
            "- Не требует специфических данных из документов\n"
            "- Может быть ответен на основе общих знаний\n\n"
            "Верни две строки строго в формате:\n"
            "Decision: <YES|NO>\n"
            "Reason: <краткое техническое обоснование>\n\n"
            "Вопрос: {question}"
        ),
        input_variables=["question"],
    )

    rewrite_prompt = PromptTemplate(
        template=(
            "Ты — эксперт по техническому поиску в нормативных документах.\n"
            "Переформулируй вопрос для максимально эффективного поиска по строительным нормам и техническим документам.\n\n"
            "Включи:\n"
            "- Ключевые технические термины\n"
            "- Номера разделов, пунктов, таблиц (если упоминаются)\n"
            "- Синонимы и альтернативные формулировки\n"
            "- Специфические единицы измерения\n"
            "- Типы конструкций, материалов, процессов\n\n"
            "Выведи только переформулированный технический запрос одной строкой без пояснений.\n\n"
            "Вопрос: {question}"
        ),
        input_variables=["question"],
    )

    qa_prompt = PromptTemplate(
        template=(
            "Ты — эксперт по строительным нормам и техническим документам Республики Казахстан.\n"
            "Ответь на вопрос, используя ТОЛЬКО предоставленные фрагменты нормативных документов.\n"
            "Строго придерживайся технической точности и не добавляй информацию из внешних источников.\n\n"
            "КРИТИЧЕСКИ ВАЖНО:\n"
            "- Используй ТОЛЬКО данные из предоставленных фрагментов\n"
            "- Указывай точные номера пунктов, разделов, таблиц из документов\n"
            "- Сохраняй техническую терминологию и единицы измерения\n"
            "- В разделе 'Источники' указывай ТОЛЬКО реальные источники из контекста\n"
            "- НЕ выдумывай названия документов, номера страниц или технические параметры\n\n"
            "Контекст:\n{context}\n\n"
            "Вопрос: {question}\n\n"
            "Структура ответа:\n"
            "1. **Прямой ответ** - четкий технический ответ на вопрос\n"
            "2. **Технические детали** - конкретные параметры, нормы, требования\n"
            "3. **Нормативные ссылки** - номера пунктов, разделов, таблиц\n"
            "4. **Источники** - ТОЛЬКО реальные источники из контекста выше\n\n"
            "Ответ:\n"
            "Развернутый технический ответ:\n"
            "Источники: (укажи ТОЛЬКО реальные источники из контекста выше)"
        ),
        input_variables=["context", "question"],
    )

    critique_prompt = PromptTemplate(
        template=(
            "Ты — эксперт по техническому анализу нормативных документов.\n"
            "Проанализируй черновой ответ относительно технического вопроса и приведённого контекста нормативных документов.\n\n"
            "Оцени:\n"
            "- Достаточность технических данных для полного ответа\n"
            "- Необходимость дополнительного поиска по специфическим терминам\n"
            "- Риск технических неточностей или галлюцинаций\n"
            "- Качество ссылок на нормативные документы\n\n"
            "Сформируй строгую техническую сводку в формате строк:\n"
            "Evidence_Sufficiency: <YES|NO>\n"
            "Need_More_Retrieval: <YES|NO>\n"
            "Hallucination_Risk: <LOW|MEDIUM|HIGH>\n"
            "Suggested_Terms: <технические термины для дополнительного поиска или '-' >\n"
            "Notes: <технические рекомендации по улучшению ответа>\n\n"
            "Вопрос: {question}\n\n"
            "Черновой ответ: {draft}\n\n"
            "Контекст:\n{context}"
        ),
        input_variables=["question", "draft", "context"],
    )

    revise_prompt = PromptTemplate(
        template=(
            "Ты — эксперт по строительным нормам. Улучши технический ответ с учётом дополнительного контекста и замечаний самокритики.\n\n"
            "Сохрани техническую структуру:\n"
            "1. **Прямой ответ** - четкий технический ответ\n"
            "2. **Технические детали** - конкретные параметры и нормы\n"
            "3. **Нормативные ссылки** - точные номера пунктов и разделов\n"
            "4. **Источники** - ТОЛЬКО реальные источники из контекста\n\n"
            "Учти все технические рекомендации и обеспечь максимальную точность.\n\n"
            "Вопрос: {question}\n\n"
            "Замечания самокритики:\n{critique}\n\n"
            "Контекст:\n{context}\n\n"
            "Черновой ответ:\n{draft}"
        ),
        input_variables=["question", "critique", "context", "draft"],
    )

    decision_raw = llm.invoke(decide_prompt.format(question=question)).content
    decision = _parse_decision(decision_raw)

    reformulated_query = llm.invoke(rewrite_prompt.format(question=question)).content.strip()

    retriever = _build_retriever(vectordb, selected_document=selected_document)
    retrieved_docs: List[Document] = []
    if decision.get("decision") != "NO":
        try:
            retrieved_docs = retriever.get_relevant_documents(reformulated_query)
        except Exception:
            retrieved_docs = []

    context_text = _format_context(_uniq_documents(retrieved_docs, max_chunks))

    draft_answer = llm.invoke(
        qa_prompt.format(context=context_text, question=question)
    ).content.strip()

    critique_raw = llm.invoke(
        critique_prompt.format(question=question, draft=draft_answer, context=context_text)
    ).content.strip()
    critique = _parse_critique(critique_raw)

    final_answer = draft_answer

    try_another_round = (
        critique.get("need_more_retrieval") == "YES" and max_rounds > 1
    )

    if try_another_round:
        additional_terms = critique.get("suggested_terms", "").strip()
        if additional_terms and additional_terms != "-":
            augmented_query = f"{reformulated_query} {additional_terms}"
        else:
            augmented_query = reformulated_query

        try:
            more_docs = retriever.get_relevant_documents(augmented_query)
        except Exception:
            more_docs = []

        merged_docs = _uniq_documents(retrieved_docs + more_docs, max_chunks)
        revised_context = _format_context(merged_docs)

        final_answer = llm.invoke(
            revise_prompt.format(
                question=question,
                critique=critique_raw,
                context=revised_context,
                draft=draft_answer,
            )
        ).content.strip()

        retrieved_docs = merged_docs

    return {
        "final_answer": final_answer,
        "decision": decision,
        "reformulated_query": reformulated_query,
        "critique": critique,
        "source_documents": retrieved_docs,
    }

