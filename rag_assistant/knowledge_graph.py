"""
Knowledge Graph Builder for RAG System
Extracts entities and relationships from documents to build a knowledge graph
"""

import os
import json
import sqlite3
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from rag_assistant import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    name: str
    type: str
    description: str
    source_file: str
    page: int
    chunk_id: str

@dataclass
class Relation:
    """Represents a relationship between entities"""
    subject: str
    predicate: str
    object: str
    confidence: float
    source_file: str
    page: int
    chunk_id: str

class FileMetadata:
    """Manages file metadata table with filename, keywords, and summary"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database for file metadata"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    keywords TEXT,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def add_file_metadata(self, filename: str, keywords: str, summary: str):
        """Add or update file metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO file_metadata (filename, keywords, summary, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (filename, keywords, summary))
            conn.commit()
    
    def get_file_metadata(self, filename: str) -> Optional[Dict]:
        """Get metadata for a specific file"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT filename, keywords, summary FROM file_metadata WHERE filename = ?",
                (filename,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    'filename': row[0],
                    'keywords': row[1],
                    'summary': row[2]
                }
        return None
    
    def get_all_metadata(self) -> List[Dict]:
        """Get metadata for all files"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT filename, keywords, summary FROM file_metadata ORDER BY filename"
            )
            return [
                {
                    'filename': row[0],
                    'keywords': row[1],
                    'summary': row[2]
                }
                for row in cursor.fetchall()
            ]

class KnowledgeGraphBuilder:
    """Builds and manages knowledge graphs from documents"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=config.KG_ENTITY_EXTRACTION_MODEL,
            temperature=0,
            openai_api_key=config.API_KEY
        )
        self.graph = nx.DiGraph()
        self.file_metadata = FileMetadata(
            os.path.join(config.KNOWLEDGE_GRAPH_DIR, "file_metadata.db")
        )
        self.entities_db_path = os.path.join(config.KNOWLEDGE_GRAPH_DIR, "entities.json")
        self.relations_db_path = os.path.join(config.KNOWLEDGE_GRAPH_DIR, "relations.json")
        
        # Create knowledge graph directory
        os.makedirs(config.KNOWLEDGE_GRAPH_DIR, exist_ok=True)
    
    def _load_existing_data(self):
        """Load existing entities and relations from disk"""
        try:
            if os.path.exists(self.entities_db_path):
                with open(self.entities_db_path, 'r', encoding='utf-8') as f:
                    entities_data = json.load(f)
                    for entity_data in entities_data:
                        entity = Entity(**entity_data)
                        self.graph.add_node(
                            entity.name,
                            type=entity.type,
                            description=entity.description,
                            source_file=entity.source_file,
                            page=entity.page,
                            chunk_id=entity.chunk_id
                        )
            
            if os.path.exists(self.relations_db_path):
                with open(self.relations_db_path, 'r', encoding='utf-8') as f:
                    relations_data = json.load(f)
                    for relation_data in relations_data:
                        relation = Relation(**relation_data)
                        self.graph.add_edge(
                            relation.subject,
                            relation.object,
                            predicate=relation.predicate,
                            confidence=relation.confidence,
                            source_file=relation.source_file,
                            page=relation.page,
                            chunk_id=relation.chunk_id
                        )
            
            logger.info(f"Loaded {len(self.graph.nodes)} entities and {len(self.graph.edges)} relations")
        except Exception as e:
            logger.warning(f"Could not load existing knowledge graph data: {e}")
    
    def _save_data(self):
        """Save entities and relations to disk"""
        try:
            # Save entities
            entities_data = []
            for node, data in self.graph.nodes(data=True):
                entities_data.append({
                    'name': node,
                    'type': data.get('type', 'unknown'),
                    'description': data.get('description', ''),
                    'source_file': data.get('source_file', ''),
                    'page': data.get('page', 0),
                    'chunk_id': data.get('chunk_id', '')
                })
            
            with open(self.entities_db_path, 'w', encoding='utf-8') as f:
                json.dump(entities_data, f, ensure_ascii=False, indent=2)
            
            # Save relations
            relations_data = []
            for edge in self.graph.edges(data=True):
                relations_data.append({
                    'subject': edge[0],
                    'predicate': edge[2].get('predicate', ''),
                    'object': edge[1],
                    'confidence': edge[2].get('confidence', 0.0),
                    'source_file': edge[2].get('source_file', ''),
                    'page': edge[2].get('page', 0),
                    'chunk_id': edge[2].get('chunk_id', '')
                })
            
            with open(self.relations_db_path, 'w', encoding='utf-8') as f:
                json.dump(relations_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(entities_data)} entities and {len(relations_data)} relations")
        except Exception as e:
            logger.error(f"Error saving knowledge graph data: {e}")
    
    def extract_entities(self, text: str, source_file: str, page: int, chunk_id: str) -> List[Entity]:
        """Extract entities from text using LLM"""
        prompt = PromptTemplate(
            template="""
            Ты - эксперт по извлечению сущностей из технических документов по строительным нормам.
            
            Извлеки из текста все важные сущности (не более {max_entities}):
            - Технические термины и понятия
            - Номера разделов, пунктов, таблиц
            - Типы конструкций, материалов
            - Единицы измерения
            - Нормативные значения
            - Процессы и процедуры
            
            Для каждой сущности укажи:
            1. Название сущности
            2. Тип (TERM, SECTION, MATERIAL, VALUE, UNIT, PROCESS, etc.)
            3. Краткое описание
            
            Формат ответа (JSON):
            [
                {{"name": "название", "type": "тип", "description": "описание"}},
                ...
            ]
            
            Текст: {text}
            """,
            input_variables=["text", "max_entities"]
        )
        
        try:
            response = self.llm.invoke(
                prompt.format(text=text, max_entities=config.KG_MAX_ENTITIES_PER_CHUNK)
            ).content.strip()
            
            # Parse JSON response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                entities_data = json.loads(json_match.group())
                entities = []
                for entity_data in entities_data:
                    entity = Entity(
                        name=entity_data['name'],
                        type=entity_data['type'],
                        description=entity_data['description'],
                        source_file=source_file,
                        page=page,
                        chunk_id=chunk_id
                    )
                    entities.append(entity)
                return entities
        except Exception as e:
            logger.warning(f"Error extracting entities: {e}")
        
        return []
    
    def extract_relations(self, text: str, entities: List[Entity], source_file: str, page: int, chunk_id: str) -> List[Relation]:
        """Extract relationships between entities"""
        if len(entities) < 2:
            return []
        
        entity_names = [e.name for e in entities]
        
        prompt = PromptTemplate(
            template="""
            Ты - эксперт по извлечению отношений из технических документов.
            
            Найди отношения между сущностями в тексте (не более {max_relations}):
            - Определения и описания
            - Зависимости и связи
            - Иерархии и классификации
            - Количественные соотношения
            - Процедурные связи
            
            Формат ответа (JSON):
            [
                {{"subject": "субъект", "predicate": "отношение", "object": "объект", "confidence": 0.9}},
                ...
            ]
            
            Сущности: {entities}
            Текст: {text}
            """,
            input_variables=["entities", "text", "max_relations"]
        )
        
        try:
            response = self.llm.invoke(
                prompt.format(
                    entities=entity_names,
                    text=text,
                    max_relations=config.KG_MAX_RELATIONS_PER_CHUNK
                )
            ).content.strip()
            
            # Parse JSON response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                relations_data = json.loads(json_match.group())
                relations = []
                for rel_data in relations_data:
                    relation = Relation(
                        subject=rel_data['subject'],
                        predicate=rel_data['predicate'],
                        object=rel_data['object'],
                        confidence=rel_data.get('confidence', 0.5),
                        source_file=source_file,
                        page=page,
                        chunk_id=chunk_id
                    )
                    relations.append(relation)
                return relations
        except Exception as e:
            logger.warning(f"Error extracting relations: {e}")
        
        return []
    
    def generate_file_summary(self, filename: str, documents: List[Document]) -> Tuple[str, str]:
        """Generate keywords and summary for a file"""
        # Combine all text from the file
        file_text = "\n".join([doc.page_content for doc in documents if doc.metadata.get('source') == filename])
        
        if not file_text.strip():
            return "", ""
        
        summary_prompt = PromptTemplate(
            template="""
            Ты - эксперт по анализу технических документов по строительным нормам.
            
            Проанализируй документ и создай:
            1. Ключевые слова (через запятую, не более 20)
            2. Краткое содержание (2-3 предложения)
            
            Формат ответа:
            Ключевые слова: ключ1, ключ2, ключ3...
            Краткое содержание: Описание основных тем и содержания документа.
            
            Документ: {text}
            """,
            input_variables=["text"]
        )
        
        try:
            response = self.llm.invoke(summary_prompt.format(text=file_text[:8000])).content.strip()
            
            keywords = ""
            summary = ""
            
            lines = response.split('\n')
            for line in lines:
                if line.startswith('Ключевые слова:'):
                    keywords = line.replace('Ключевые слова:', '').strip()
                elif line.startswith('Краткое содержание:'):
                    summary = line.replace('Краткое содержание:', '').strip()
            
            return keywords, summary
        except Exception as e:
            logger.warning(f"Error generating file summary: {e}")
            return "", ""
    
    def build_from_documents(self, documents: List[Document], recreate: bool = False):
        """Build knowledge graph from documents"""
        if recreate:
            logger.info("Recreating knowledge graph...")
            self.graph.clear()
            # Clear existing files
            for file_path in [self.entities_db_path, self.relations_db_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
        else:
            self._load_existing_data()
        
        # Group documents by source file
        files_data = {}
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            if source not in files_data:
                files_data[source] = []
            files_data[source].append(doc)
        
        # Process each file
        for filename, file_docs in files_data.items():
            logger.info(f"Processing file: {filename}")
            
            # Generate file metadata
            keywords, summary = self.generate_file_summary(filename, file_docs)
            self.file_metadata.add_file_metadata(filename, keywords, summary)
            
            # Process each document chunk
            for doc in file_docs:
                chunk_id = f"{filename}_{doc.metadata.get('page', 0)}"
                
                # Extract entities
                entities = self.extract_entities(
                    doc.page_content,
                    filename,
                    doc.metadata.get('page', 0),
                    chunk_id
                )
                
                # Add entities to graph
                for entity in entities:
                    self.graph.add_node(
                        entity.name,
                        type=entity.type,
                        description=entity.description,
                        source_file=entity.source_file,
                        page=entity.page,
                        chunk_id=entity.chunk_id
                    )
                
                # Extract relations
                relations = self.extract_relations(
                    doc.page_content,
                    entities,
                    filename,
                    doc.metadata.get('page', 0),
                    chunk_id
                )
                
                # Add relations to graph
                for relation in relations:
                    self.graph.add_edge(
                        relation.subject,
                        relation.object,
                        predicate=relation.predicate,
                        confidence=relation.confidence,
                        source_file=relation.source_file,
                        page=relation.page,
                        chunk_id=relation.chunk_id
                    )
        
        # Save the knowledge graph
        self._save_data()
        
        logger.info(f"Knowledge graph built: {len(self.graph.nodes)} entities, {len(self.graph.edges)} relations")
    
    def get_related_entities(self, entity_name: str, max_depth: int = 2) -> List[str]:
        """Get entities related to the given entity"""
        if entity_name not in self.graph:
            return []
        
        related = set()
        current_level = {entity_name}
        
        for depth in range(max_depth):
            next_level = set()
            for entity in current_level:
                # Get neighbors
                neighbors = list(self.graph.neighbors(entity))
                related.update(neighbors)
                next_level.update(neighbors)
            current_level = next_level - related
        
        return list(related)
    
    def get_entity_context(self, entity_name: str) -> Dict:
        """Get context information for an entity"""
        if entity_name not in self.graph:
            return {}
        
        node_data = self.graph.nodes[entity_name]
        neighbors = list(self.graph.neighbors(entity_name))
        
        return {
            'entity': entity_name,
            'type': node_data.get('type', 'unknown'),
            'description': node_data.get('description', ''),
            'source_file': node_data.get('source_file', ''),
            'page': node_data.get('page', 0),
            'related_entities': neighbors,
            'relations': [
                {
                    'target': neighbor,
                    'predicate': self.graph[entity_name][neighbor].get('predicate', ''),
                    'confidence': self.graph[entity_name][neighbor].get('confidence', 0.0)
                }
                for neighbor in neighbors
            ]
        }