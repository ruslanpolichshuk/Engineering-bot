# Self-RAG Implementation with LangGraph

## Overview

This implementation provides a comprehensive Self-RAG (Self-Reflective Retrieval-Augmented Generation) system using LangGraph for workflow orchestration. The system implements all the specific token types requested and includes accuracy percentage metrics.

## Key Features Implemented

### 1. Self-RAG Tokens

#### Retrieve Token
- **Input**: x (question) OR x (question), y (generation)
- **Output**: yes, no, continue
- **Purpose**: Decides whether to retrieve D chunks based on the question or question+generation

#### ISREL Token
- **Input**: (x (question), d (chunk)) for d in D
- **Output**: relevant, irrelevant
- **Purpose**: Decides whether passages D are relevant to the question

#### ISSUP Token
- **Input**: x (question), d (chunk), y (generation) for d in D
- **Output**: fully supported, partially supported, no support
- **Purpose**: Decides whether LLM generation from each chunk is relevant to the chunk

#### ISUSE Token
- **Input**: x (question), y (generation) for d in D
- **Output**: {5, 4, 3, 2, 1}
- **Purpose**: Decides whether generation from each chunk is useful response to the question

### 2. Accuracy Percentage Metric

The system calculates accuracy percentage based on:
- **Relevance scores** (30% weight): Percentage of relevant documents
- **Support scores** (40% weight): Quality of support for generated answer
- **Usefulness scores** (30% weight): Overall usefulness rating

### 3. Technical Improvements

#### Enhanced Prompts
- All prompts are now focused on technical accuracy
- Emphasis on documentation-only responses
- Clear instructions to avoid hallucinations
- Structured output format for technical answers

#### LangGraph Workflow
- Proper state management with TypedDict
- Sequential node execution
- Error handling and fallbacks
- Modular design for easy extension

## File Structure

```
/workspace/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Updated with LangGraph dependency
├── rag_assistant/
│   ├── config.py                  # Configuration settings
│   ├── main.py                    # Original RAG implementation
│   ├── self_rag.py                # Original Self-RAG implementation
│   ├── self_rag_langgraph.py      # New LangGraph-based Self-RAG
│   └── utils.py                   # Utility functions
└── downloaded_pdfs/               # PDF documents directory
```

## Usage

The application now provides three RAG modes:

1. **Обычный RAG** - Standard retrieval-augmented generation
2. **Self-RAG (старый)** - Original Self-RAG implementation
3. **Self-RAG LangGraph (новый)** - New LangGraph-based implementation (default)

## Key Implementation Details

### SelfRAGState
```python
class SelfRAGState(TypedDict):
    question: str
    generation: Optional[str]
    retrieved_docs: List[Document]
    retrieve_decision: str
    relevance_scores: Dict[str, str]
    support_scores: Dict[str, str]
    usefulness_scores: Dict[str, int]
    accuracy_percentage: float
    final_answer: str
    selected_document: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]
```

### Workflow Nodes
1. `retrieve_decision` - Determines if retrieval is needed
2. `retrieve_documents` - Retrieves relevant documents
3. `assess_relevance` - Evaluates document relevance
4. `generate_answer` - Generates technical answer
5. `assess_support` - Evaluates answer support
6. `assess_usefulness` - Evaluates answer usefulness
7. `calculate_accuracy` - Calculates accuracy percentage
8. `finalize_answer` - Formats final answer with accuracy

### Technical Prompts

All prompts are designed for:
- Technical accuracy in construction norms
- Documentation-only responses
- Precise citation of sources
- Avoidance of hallucinations
- Structured technical output

## Configuration

The system uses environment variables for configuration:
- `SELF_RAG_MODEL` - LLM model (default: gpt-4o-mini)
- `SELF_RAG_TEMPERATURE` - Temperature setting (default: 0)
- `SELF_RAG_SEARCH_K` - Number of documents to retrieve (default: 10)
- `SELF_RAG_FETCH_K` - Fetch size for retrieval (default: 30)
- `SELF_RAG_SCORE_THRESHOLD` - Relevance threshold (default: 0.4)
- `SELF_RAG_MAX_CONTEXT_CHUNKS` - Max context chunks (default: 10)
- `SELF_RAG_MAX_ROUNDS` - Max refinement rounds (default: 2)

## Benefits

1. **Self-Reflection**: The system evaluates its own outputs
2. **Accuracy Metrics**: Provides confidence scores for answers
3. **Technical Focus**: Optimized for construction norms and technical documents
4. **Modular Design**: Easy to extend and modify
5. **Error Handling**: Robust error handling and fallbacks
6. **Documentation-Only**: Prevents hallucinations by restricting to provided documents

## Future Enhancements

- Additional token types for specific domains
- Custom accuracy calculation methods
- Integration with external knowledge bases
- Advanced workflow branching based on token results
- Performance optimization for large document sets