from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

from rag_assistant import config


class SelfRAGState(TypedDict):
    """State for Self-RAG workflow"""
    question: str
    generation: Optional[str]
    retrieved_docs: List[Document]
    retrieve_decision: str  # yes, no, continue
    relevance_scores: Dict[str, str]  # doc_id -> relevant/irrelevant
    support_scores: Dict[str, str]  # doc_id -> fully supported/partially supported/no support
    usefulness_scores: Dict[str, int]  # doc_id -> 1-5
    accuracy_percentage: float
    final_answer: str
    selected_document: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]


class SelfRAGTokens:
    """Self-RAG token implementations"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        # Retrieve token prompt
        self.retrieve_prompt = PromptTemplate(
            template=(
                "Ты — эксперт-классификатор по строительным нормам и техническим документам.\n"
                "Определи, нужно ли обращаться к нормативным документам для получения точного технического ответа.\n\n"
                "Если вопрос касается конкретных технических требований, норм, параметров, методов расчета - ответь 'yes'\n"
                "Если вопрос общий и не требует специфических документов - ответь 'no'\n"
                "Если нужна дополнительная информация после генерации - ответь 'continue'\n\n"
                "Верни ТОЛЬКО одно слово: yes, no, или continue\n\n"
                "Вопрос: {question}\n"
                "Предыдущая генерация: {generation}\n"
            ),
            input_variables=["question", "generation"]
        )
        
        # ISREL token prompt
        self.isrel_prompt = PromptTemplate(
            template=(
                "Ты — эксперт по техническому анализу нормативных документов.\n"
                "Определи, релевантен ли данный фрагмент нормативного документа для ответа на технический вопрос.\n\n"
                "Фрагмент релевантен если содержит:\n"
                "- Прямые ответы на вопрос\n"
                "- Технические параметры, нормы, требования\n"
                "- Методы расчета, формулы, коэффициенты\n"
                "- Классификации, типы, категории\n"
                "- Процедуры, последовательности действий\n\n"
                "Верни ТОЛЬКО одно слово: relevant или irrelevant\n\n"
                "Вопрос: {question}\n"
                "Фрагмент документа: {chunk}\n"
            ),
            input_variables=["question", "chunk"]
        )
        
        # ISSUP token prompt
        self.issup_prompt = PromptTemplate(
            template=(
                "Ты — эксперт по техническому анализу нормативных документов.\n"
                "Определи, насколько хорошо данный фрагмент нормативного документа поддерживает сгенерированный технический ответ.\n\n"
                "fully supported - все утверждения в ответе прямо подтверждаются фрагментом\n"
                "partially supported - некоторые утверждения подтверждаются, есть неточности\n"
                "no support - фрагмент не подтверждает утверждения в ответе\n\n"
                "Верни ТОЛЬКО одно из слов: fully supported, partially supported, no support\n\n"
                "Вопрос: {question}\n"
                "Фрагмент документа: {chunk}\n"
                "Сгенерированный ответ: {generation}\n"
            ),
            input_variables=["question", "chunk", "generation"]
        )
        
        # ISUSE token prompt
        self.isuse_prompt = PromptTemplate(
            template=(
                "Ты — эксперт по техническому анализу нормативных документов.\n"
                "Оцени полезность сгенерированного технического ответа для данного вопроса по шкале 1-5.\n\n"
                "5 - Отличный технический ответ с точными данными и ссылками на нормы\n"
                "4 - Хороший ответ с незначительными техническими недостатками\n"
                "3 - Удовлетворительный ответ с некоторыми техническими проблемами\n"
                "2 - Неполный или неточный технический ответ\n"
                "1 - Плохой или нерелевантный технический ответ\n\n"
                "Верни ТОЛЬКО одну цифру: 5, 4, 3, 2, или 1\n\n"
                "Вопрос: {question}\n"
                "Сгенерированный ответ: {generation}\n"
            ),
            input_variables=["question", "generation"]
        )
        
        # Generation prompt
        self.generation_prompt = PromptTemplate(
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
            input_variables=["context", "question"]
        )

    def retrieve_token(self, question: str, generation: Optional[str] = None) -> str:
        """Retrieve token: decides to retrieve D chunks with input x (question) OR x (question), y (generation)"""
        if generation is None:
            generation = ""
        
        response = self.llm.invoke(
            self.retrieve_prompt.format(question=question, generation=generation)
        ).content.strip().lower()
        
        if response in ["yes", "no", "continue"]:
            return response
        elif "yes" in response:
            return "yes"
        elif "no" in response:
            return "no"
        elif "continue" in response:
            return "continue"
        else:
            return "yes"  # Default to yes for safety

    def isrel_token(self, question: str, chunk: str) -> str:
        """ISREL token: decides whether passages D are relevant to x with input (x (question), d (chunk))"""
        response = self.llm.invoke(
            self.isrel_prompt.format(question=question, chunk=chunk)
        ).content.strip().lower()
        
        if "relevant" in response:
            return "relevant"
        elif "irrelevant" in response:
            return "irrelevant"
        else:
            return "relevant"  # Default to relevant for safety

    def issup_token(self, question: str, chunk: str, generation: str) -> str:
        """ISSUP token: decides whether LLM generation from each chunk in D is relevant to the chunk"""
        response = self.llm.invoke(
            self.issup_prompt.format(question=question, chunk=chunk, generation=generation)
        ).content.strip().lower()
        
        if "fully supported" in response:
            return "fully supported"
        elif "partially supported" in response:
            return "partially supported"
        elif "no support" in response:
            return "no support"
        else:
            return "partially supported"  # Default

    def isuse_token(self, question: str, generation: str) -> int:
        """ISUSE token: decides whether generation from each chunk in D is useful response to x"""
        response = self.llm.invoke(
            self.isuse_prompt.format(question=question, generation=generation)
        ).content.strip()
        
        try:
            score = int(response)
            if 1 <= score <= 5:
                return score
        except ValueError:
            pass
        
        # Try to extract number from response
        for i in range(1, 6):
            if str(i) in response:
                return i
        
        return 3  # Default to middle score

    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using context"""
        return self.llm.invoke(
            self.generation_prompt.format(context=context, question=question)
        ).content.strip()


def _build_retriever(vectordb, selected_document: Optional[str] = None):
    """Build retriever with appropriate settings"""
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
    """Format documents into context string"""
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


def _calculate_accuracy_percentage(relevance_scores: Dict[str, str], 
                                 support_scores: Dict[str, str], 
                                 usefulness_scores: Dict[str, int]) -> float:
    """Calculate accuracy percentage based on token assessments"""
    if not relevance_scores and not support_scores and not usefulness_scores:
        return 0.0
    
    total_score = 0.0
    total_weight = 0.0
    
    # Relevance weight: 30%
    if relevance_scores:
        relevant_count = sum(1 for score in relevance_scores.values() if score == "relevant")
        relevance_percentage = (relevant_count / len(relevance_scores)) * 100
        total_score += relevance_percentage * 0.3
        total_weight += 0.3
    
    # Support weight: 40%
    if support_scores:
        support_values = {"fully supported": 100, "partially supported": 60, "no support": 0}
        support_percentage = sum(support_values.get(score, 0) for score in support_scores.values()) / len(support_scores)
        total_score += support_percentage * 0.4
        total_weight += 0.4
    
    # Usefulness weight: 30%
    if usefulness_scores:
        usefulness_percentage = (sum(usefulness_scores.values()) / len(usefulness_scores)) * 20  # Convert 1-5 to 0-100
        total_score += usefulness_percentage * 0.3
        total_weight += 0.3
    
    if total_weight == 0:
        return 0.0
    
    return round(total_score / total_weight, 1)


def create_self_rag_workflow(vectordb) -> StateGraph:
    """Create Self-RAG workflow using LangGraph"""
    
    llm = ChatOpenAI(
        model_name=getattr(config, "SELF_RAG_MODEL", "gpt-4o-mini"),
        temperature=getattr(config, "SELF_RAG_TEMPERATURE", 0),
        openai_api_key=config.API_KEY,
    )
    
    tokens = SelfRAGTokens(llm)
    
    def retrieve_decision_node(state: SelfRAGState) -> SelfRAGState:
        """Retrieve decision node"""
        decision = tokens.retrieve_token(state["question"], state.get("generation"))
        state["retrieve_decision"] = decision
        return state
    
    def retrieve_documents_node(state: SelfRAGState) -> SelfRAGState:
        """Retrieve documents node"""
        if state["retrieve_decision"] in ["yes", "continue"]:
            retriever = _build_retriever(vectordb, state.get("selected_document"))
            try:
                retrieved_docs = retriever.get_relevant_documents(state["question"])
                state["retrieved_docs"] = retrieved_docs
            except Exception:
                state["retrieved_docs"] = []
        else:
            state["retrieved_docs"] = []
        return state
    
    def assess_relevance_node(state: SelfRAGState) -> SelfRAGState:
        """Assess relevance of retrieved documents"""
        relevance_scores = {}
        for i, doc in enumerate(state["retrieved_docs"]):
            doc_id = f"doc_{i}"
            relevance = tokens.isrel_token(state["question"], doc.page_content)
            relevance_scores[doc_id] = relevance
        state["relevance_scores"] = relevance_scores
        return state
    
    def generate_answer_node(state: SelfRAGState) -> SelfRAGState:
        """Generate answer node"""
        if state["retrieved_docs"]:
            context = _format_context(state["retrieved_docs"])
            generation = tokens.generate_answer(state["question"], context)
            state["generation"] = generation
        else:
            state["generation"] = "Недостаточно информации для ответа на вопрос."
        return state
    
    def assess_support_node(state: SelfRAGState) -> SelfRAGState:
        """Assess support for generated answer"""
        support_scores = {}
        for i, doc in enumerate(state["retrieved_docs"]):
            doc_id = f"doc_{i}"
            support = tokens.issup_token(state["question"], doc.page_content, state["generation"])
            support_scores[doc_id] = support
        state["support_scores"] = support_scores
        return state
    
    def assess_usefulness_node(state: SelfRAGState) -> SelfRAGState:
        """Assess usefulness of generated answer"""
        usefulness_scores = {}
        usefulness = tokens.isuse_token(state["question"], state["generation"])
        for i in range(len(state["retrieved_docs"])):
            doc_id = f"doc_{i}"
            usefulness_scores[doc_id] = usefulness
        state["usefulness_scores"] = usefulness_scores
        return state
    
    def calculate_accuracy_node(state: SelfRAGState) -> SelfRAGState:
        """Calculate accuracy percentage"""
        accuracy = _calculate_accuracy_percentage(
            state["relevance_scores"],
            state["support_scores"],
            state["usefulness_scores"]
        )
        state["accuracy_percentage"] = accuracy
        return state
    
    def finalize_answer_node(state: SelfRAGState) -> SelfRAGState:
        """Finalize answer with accuracy percentage"""
        accuracy = state["accuracy_percentage"]
        generation = state["generation"]
        
        final_answer = f"**Точность ответа: {accuracy}%**\n\n{generation}"
        state["final_answer"] = final_answer
        return state
    
    # Create workflow
    workflow = StateGraph(SelfRAGState)
    
    # Add nodes
    workflow.add_node("retrieve_decision", retrieve_decision_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("assess_relevance", assess_relevance_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("assess_support", assess_support_node)
    workflow.add_node("assess_usefulness", assess_usefulness_node)
    workflow.add_node("calculate_accuracy", calculate_accuracy_node)
    workflow.add_node("finalize_answer", finalize_answer_node)
    
    # Add edges
    workflow.add_edge(START, "retrieve_decision")
    workflow.add_edge("retrieve_decision", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "assess_relevance")
    workflow.add_edge("assess_relevance", "generate_answer")
    workflow.add_edge("generate_answer", "assess_support")
    workflow.add_edge("assess_support", "assess_usefulness")
    workflow.add_edge("assess_usefulness", "calculate_accuracy")
    workflow.add_edge("calculate_accuracy", "finalize_answer")
    workflow.add_edge("finalize_answer", END)
    
    return workflow


def run_self_rag_langgraph(vectordb, question: str, selected_document: Optional[str] = None) -> Dict:
    """Run Self-RAG workflow using LangGraph"""
    
    workflow = create_self_rag_workflow(vectordb)
    app = workflow.compile()
    
    initial_state = {
        "question": question,
        "generation": None,
        "retrieved_docs": [],
        "retrieve_decision": "",
        "relevance_scores": {},
        "support_scores": {},
        "usefulness_scores": {},
        "accuracy_percentage": 0.0,
        "final_answer": "",
        "selected_document": selected_document,
        "messages": []
    }
    
    result = app.invoke(initial_state)
    
    return {
        "final_answer": result["final_answer"],
        "accuracy_percentage": result["accuracy_percentage"],
        "retrieve_decision": result["retrieve_decision"],
        "relevance_scores": result["relevance_scores"],
        "support_scores": result["support_scores"],
        "usefulness_scores": result["usefulness_scores"],
        "source_documents": result["retrieved_docs"],
        "generation": result["generation"]
    }