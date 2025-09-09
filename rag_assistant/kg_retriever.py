"""
Knowledge Graph Enhanced Retriever
Combines vector similarity search with knowledge graph traversal for better retrieval
"""

import os
from typing import List, Dict, Tuple, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from rag_assistant.knowledge_graph import KnowledgeGraphBuilder
from rag_assistant import config
import logging

logger = logging.getLogger(__name__)

class KnowledgeGraphRetriever:
    """Enhanced retriever that combines vector search with knowledge graph"""
    
    def __init__(self, vectordb: Chroma, kg_builder: KnowledgeGraphBuilder):
        self.vectordb = vectordb
        self.kg_builder = kg_builder
        self.llm = ChatOpenAI(
            model_name=config.SELF_RAG_MODEL,
            temperature=0,
            openai_api_key=config.API_KEY
        )
    
    def extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from the query"""
        prompt = PromptTemplate(
            template="""
            Ты - эксперт по извлечению ключевых терминов из технических запросов.
            
            Извлеки из запроса все важные технические термины и понятия:
            - Технические термины
            - Номера разделов, пунктов
            - Типы конструкций, материалов
            - Единицы измерения
            - Нормативные значения
            
            Верни список терминов через запятую (не более 10):
            
            Запрос: {query}
            """,
            input_variables=["query"]
        )
        
        try:
            response = self.llm.invoke(prompt.format(query=query)).content.strip()
            # Extract terms from response
            terms = [term.strip() for term in response.split(',') if term.strip()]
            return terms[:10]  # Limit to 10 terms
        except Exception as e:
            logger.warning(f"Error extracting query entities: {e}")
            return []
    
    def get_kg_enhanced_documents(self, query: str, k: int = 10) -> List[Document]:
        """Get documents enhanced with knowledge graph context"""
        # Extract entities from query
        query_entities = self.extract_query_entities(query)
        
        # Get related entities from knowledge graph
        related_entities = set()
        for entity in query_entities:
            related = self.kg_builder.get_related_entities(entity, max_depth=2)
            related_entities.update(related)
        
        # Combine original query with related entities
        enhanced_query = query
        if related_entities:
            enhanced_query += " " + " ".join(list(related_entities)[:5])  # Limit to 5 related entities
        
        # Perform vector search with enhanced query
        try:
            retriever = self.vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": k * 3,
                    "score_threshold": config.SELF_RAG_SCORE_THRESHOLD
                }
            )
            documents = retriever.get_relevant_documents(enhanced_query)
        except Exception as e:
            logger.warning(f"Error in vector search: {e}")
            # Fallback to original query
            try:
                retriever = self.vectordb.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": k,
                        "fetch_k": k * 3,
                        "score_threshold": config.SELF_RAG_SCORE_THRESHOLD
                    }
                )
                documents = retriever.get_relevant_documents(query)
            except Exception as e2:
                logger.error(f"Fallback vector search also failed: {e2}")
                documents = []
        
        # Enhance documents with knowledge graph context
        enhanced_documents = []
        for doc in documents:
            enhanced_doc = self._enhance_document_with_kg(doc, query_entities)
            enhanced_documents.append(enhanced_doc)
        
        return enhanced_documents
    
    def _enhance_document_with_kg(self, doc: Document, query_entities: List[str]) -> Document:
        """Enhance a document with knowledge graph context"""
        # Get file metadata
        source_file = doc.metadata.get('source', '')
        file_metadata = self.kg_builder.file_metadata.get_file_metadata(source_file)
        
        # Create enhanced content
        enhanced_content = doc.page_content
        
        # Add file summary if available
        if file_metadata and file_metadata.get('summary'):
            enhanced_content = f"[Файл: {source_file}]\n{file_metadata['summary']}\n\n{enhanced_content}"
        
        # Add knowledge graph context for relevant entities
        kg_context = []
        for entity in query_entities:
            if entity in self.kg_builder.graph:
                context = self.kg_builder.get_entity_context(entity)
                if context:
                    kg_context.append(f"Контекст {entity}: {context['description']}")
        
        if kg_context:
            enhanced_content += f"\n\n[Контекст из базы знаний]:\n" + "\n".join(kg_context)
        
        # Create enhanced document
        enhanced_doc = Document(
            page_content=enhanced_content,
            metadata={
                **doc.metadata,
                'file_keywords': file_metadata.get('keywords', '') if file_metadata else '',
                'file_summary': file_metadata.get('summary', '') if file_metadata else '',
                'kg_enhanced': True
            }
        )
        
        return enhanced_doc
    
    def get_documents_with_citations(self, query: str, k: int = 10) -> Tuple[List[Document], List[Dict]]:
        """Get documents with proper citations using file metadata"""
        documents = self.get_kg_enhanced_documents(query, k)
        
        citations = []
        for doc in documents:
            source_file = doc.metadata.get('source', '')
            page = doc.metadata.get('page', 0)
            
            # Get file metadata for citation
            file_metadata = self.kg_builder.file_metadata.get_file_metadata(source_file)
            
            citation = {
                'source': source_file,
                'page': page,
                'keywords': file_metadata.get('keywords', '') if file_metadata else '',
                'summary': file_metadata.get('summary', '') if file_metadata else '',
                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            citations.append(citation)
        
        return documents, citations