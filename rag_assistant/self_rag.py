from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain.schema import Document, SystemMessage, HumanMessage

from rag_assistant import config


def _call_json_llm(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
	"""
	Best-effort JSON response from LLM. Falls back to minimal structure if parsing fails.
	"""
	try:
		model = llm.bind(response_format={"type": "json_object"})
	except Exception:
		model = llm

	messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
	resp = model.invoke(messages)  # type: ignore[attr-defined]
	text = getattr(resp, "content", "")
	try:
		return json.loads(text)
	except Exception:
		# Attempt to extract JSON substring
		start = text.find("{")
		end = text.rfind("}")
		if start != -1 and end != -1:
			candidate = text[start : end + 1]
			try:
				return json.loads(candidate)
			except Exception:
				pass
		return {"text": text}


def _decide_retrieval(llm: ChatOpenAI, question: str) -> Tuple[bool, str]:
	"""Return (should_retrieve, reason)."""
	sys_prompt = (
		"You are a retrieval controller. Decide if the question requires external documents. "
		"Return strict JSON with keys: should_retrieve (boolean), reason (string)."
	)
	usr_prompt = (
		"Question: " + question + "\n"
		"Consider whether specialized, factual, or up-to-date information likely exists in provided PDFs."
	)
	obj = _call_json_llm(llm, sys_prompt, usr_prompt)
	should = bool(obj.get("should_retrieve", True))
	reason = str(obj.get("reason", "Retrieval likely beneficial for grounded answer."))
	return should, reason


def _retrieve(
	vectordb: Any,
	question: str,
	selected_document: Optional[str],
	k: int = 8,
) -> List[Tuple[Document, float]]:
	"""
	Return list of (Document, score). Uses Chroma's similarity_search_with_score with optional filter by source.
	"""
	kwargs: Dict[str, Any] = {"k": k}
	if selected_document:
		kwargs["filter"] = {"source": selected_document}
	try:
		results = vectordb.similarity_search_with_score(question, **kwargs)
	except TypeError:
		# Fallback: search without filter then post-filter
		results = vectordb.similarity_search_with_score(question, k=k)
		if selected_document:
			results = [(d, s) for d, s in results if d.metadata.get("source") == selected_document]
	return results


def _format_context(docs: List[Tuple[Document, float]]) -> Tuple[str, List[Dict[str, Any]]]:
	"""Create a readable context block and a list of citation dicts with ids."""
	lines: List[str] = []
	cites: List[Dict[str, Any]] = []
	for idx, (doc, score) in enumerate(docs, start=1):
		doc_id = f"D{idx}"
		source = str(doc.metadata.get("source", "unknown"))
		page = int(doc.metadata.get("page", -1))
		lines.append(
			f"[{doc_id}] source={source} page={page} score={score:.4f}\n{doc.page_content}"
		)
		cites.append({"doc_id": doc_id, "source": source, "page": page, "score": float(score)})
	return "\n\n".join(lines), cites


def run_self_rag(
	question: str,
	vectordb: Any,
	selected_document: Optional[str] = None,
	model_name: str = "gpt-4o",
) -> Dict[str, Any]:
	"""
	Self-RAG style pipeline:
	1) Controller decides if retrieval is needed
	2) If needed, retrieve contexts
	3) Single-pass grounded generation with self-critique and structured JSON
	Returns a dict with keys: answer, used_retrieval, retrieve_reason, citations, critique
	"""
	llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=config.API_KEY)

	should_retrieve, reason = _decide_retrieval(llm, question)
	used_retrieval = should_retrieve
	context_block = ""
	citation_meta: List[Dict[str, Any]] = []
	if should_retrieve:
		pairs = _retrieve(vectordb, question, selected_document, k=10)
		context_block, citation_meta = _format_context(pairs)

	sys_prompt = (
		"You are a Self-RAG generator. If contexts are provided, answer ONLY using them and cite doc ids. "
		"If no context is provided, answer from parametric knowledge with clear uncertainty. "
		"Return strict JSON with keys: answer (string), citations (array of objects with doc_id, quote), "
		"confidence (number 0-1), critique (object with groundedness:boolean, completeness:integer 1-5, "
		"harmfulness_risk:string in [low,medium,high], comments:string)."
	)

	usr_prompt = []
	if context_block:
		usr_prompt.append("Context documents:\n" + context_block)
	usr_prompt.append("Question: " + question)
	usr_prompt.append("Instructions: Keep the answer concise. Use [doc_id] markers in-text.")
	usr_text = "\n\n".join(usr_prompt)

	obj = _call_json_llm(llm, sys_prompt, usr_text)
	answer = str(obj.get("answer", obj.get("text", ""))).strip()
	confidence = obj.get("confidence")
	# Normalize citations
	citations_out: List[Dict[str, Any]] = []
	if isinstance(obj.get("citations"), list):
		for c in obj["citations"]:
			if isinstance(c, dict) and "doc_id" in c:
				citations_out.append({
					"doc_id": c.get("doc_id"),
					"quote": c.get("quote", ""),
				})

	critique = obj.get("critique", {})
	if not isinstance(critique, dict):
		critique = {"comments": str(critique)}

	return {
		"answer": answer,
		"used_retrieval": used_retrieval,
		"retrieve_reason": reason,
		"citations": citations_out,
		"citation_meta": citation_meta,
		"critique": critique,
		"confidence": confidence,
	}

