# Enhanced KG-RAG System with GPT-5

## Overview

This enhanced RAG (Retrieval-Augmented Generation) system integrates knowledge graphs with vector-based retrieval to provide superior document understanding and question answering capabilities. The system now uses GPT-5 and includes advanced features for building and utilizing knowledge graphs.

## Key Features

### 🧠 Knowledge Graph Integration
- **Entity Extraction**: Automatically extracts technical terms, concepts, and relationships from documents
- **Relationship Mapping**: Builds a graph of connections between entities
- **Context Enhancement**: Uses graph traversal to find related information
- **Persistent Storage**: Saves knowledge graph data for reuse

### 📊 File Metadata Management
- **SQLite Database**: Stores file metadata with three columns:
  - `filename`: Exact file name
  - `keywords`: Extracted key terms
  - `summary`: Brief document summary
- **Enhanced Citations**: Uses metadata for better source attribution

### 🚀 GPT-5 Integration
- **Advanced Reasoning**: Leverages GPT-5's improved capabilities
- **Better Understanding**: Enhanced context comprehension
- **Improved Accuracy**: More precise technical responses

### 🔄 Conditional Recreation
- **RECREATE Parameter**: Set to `1` in `.env` to rebuild everything
- **Incremental Updates**: Only processes new documents by default
- **Smart Caching**: Reuses existing vector store and knowledge graph

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Documents │───▶│  Vector Store    │───▶│   RAG System    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Knowledge Graph │    │   GPT-5 LLM     │
                       └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ File Metadata    │
                       │ Database         │
                       └──────────────────┘
```

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   cp .env.template .env
   # Edit .env with your OpenAI API key
   ```

3. **Set Configuration**:
   ```env
   OPENAI_API_KEY=your_api_key_here
   SELF_RAG_MODEL=gpt-5
   RECREATE=0
   USE_KNOWLEDGE_GRAPH=true
   ```

## Usage

### Running the Application

```bash
streamlit run app.py
```

### RAG Modes

1. **KG-RAG (с графом знаний)**: Enhanced retrieval with knowledge graph
2. **Обычный RAG**: Standard vector-based retrieval with GPT-5
3. **Self-RAG (старый)**: Original self-reflective RAG
4. **Self-RAG LangGraph (новый)**: Advanced LangGraph-based Self-RAG

### Configuration Options

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RECREATE` | `0` | Set to `1` to rebuild vector store and knowledge graph |
| `USE_KNOWLEDGE_GRAPH` | `true` | Enable/disable knowledge graph features |
| `SELF_RAG_MODEL` | `gpt-5` | LLM model for generation |
| `KG_ENTITY_EXTRACTION_MODEL` | `gpt-5` | Model for entity extraction |
| `KG_MAX_ENTITIES_PER_CHUNK` | `20` | Maximum entities per document chunk |
| `KG_MAX_RELATIONS_PER_CHUNK` | `15` | Maximum relations per document chunk |

## Knowledge Graph Features

### Entity Types
- **TERM**: Technical terms and concepts
- **SECTION**: Document sections and paragraphs
- **MATERIAL**: Construction materials
- **VALUE**: Numerical values and parameters
- **UNIT**: Units of measurement
- **PROCESS**: Procedures and processes

### Relationship Types
- **DEFINES**: Definition relationships
- **REQUIRES**: Dependency relationships
- **CLASSIFIES**: Hierarchical relationships
- **MEASURES**: Quantitative relationships
- **FOLLOWS**: Procedural relationships

### File Metadata Schema

```sql
CREATE TABLE file_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT UNIQUE NOT NULL,
    keywords TEXT,
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Advanced Features

### Knowledge Graph Builder

The `KnowledgeGraphBuilder` class provides:

- **Entity Extraction**: Uses GPT-5 to identify important entities
- **Relationship Discovery**: Finds connections between entities
- **Graph Storage**: Persists graph data in JSON format
- **Context Retrieval**: Provides entity context for enhanced retrieval

### Enhanced Retriever

The `KnowledgeGraphRetriever` class offers:

- **Query Enhancement**: Expands queries with related entities
- **Context Augmentation**: Adds knowledge graph context to documents
- **Citation Enhancement**: Uses file metadata for better citations
- **Fallback Support**: Gracefully degrades to standard retrieval

### Self-RAG Integration

Enhanced Self-RAG with knowledge graph support:

- **Entity-Aware Retrieval**: Considers graph relationships
- **Context-Rich Generation**: Uses enhanced document context
- **Improved Self-Critique**: Better evaluation with graph context

## Performance Considerations

### Memory Usage
- Knowledge graphs are stored in memory during operation
- Large document collections may require significant RAM
- Consider chunking strategies for very large datasets

### Processing Time
- Initial knowledge graph construction can be time-intensive
- Entity and relation extraction uses multiple LLM calls
- Incremental updates are much faster than full rebuilds

### Storage Requirements
- Vector store: ~1-2MB per document
- Knowledge graph: ~100-500KB per document
- File metadata: ~1-5KB per document

## Troubleshooting

### Common Issues

1. **Knowledge Graph Not Loading**:
   - Check `USE_KNOWLEDGE_GRAPH=true` in `.env`
   - Verify `KNOWLEDGE_GRAPH_DIR` permissions
   - Set `RECREATE=1` to rebuild

2. **GPT-5 Not Available**:
   - Fallback to `gpt-4o` if GPT-5 is not accessible
   - Update API key and model access permissions

3. **Memory Issues**:
   - Reduce `KG_MAX_ENTITIES_PER_CHUNK`
   - Process documents in smaller batches
   - Consider using lighter embedding models

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- **Multi-modal Support**: Image and table understanding
- **Temporal Knowledge**: Time-based relationship tracking
- **Federated Graphs**: Integration with external knowledge sources
- **Real-time Updates**: Live knowledge graph updates
- **Advanced Reasoning**: Graph-based logical inference

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement enhancements
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.