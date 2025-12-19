# RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) assistant with Knowledge Graph integration and Self-RAG capabilities.

## Features

- **Knowledge Graph RAG**: Enhanced retrieval using knowledge graphs
- **Self-RAG**: Self-reflective retrieval augmented generation with LangGraph
- **Document Processing**: PDF document ingestion and processing with OCR support
- **OCR Support**: Automatic text recognition for scanned PDF documents (Tesseract OCR)
- **Vector Database**: Efficient document storage and retrieval
- **Web Interface**: Streamlit-based user interface

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git
- **Tesseract OCR** (optional, for scanned PDF processing):
  - Windows: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
  - macOS: `brew install tesseract tesseract-lang`
  - Linux: `sudo apt-get install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng poppler-utils`

## Installation

### Windows

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-assistant
   ```

2. Install Tesseract OCR (optional, for scanned PDFs):
   - Download installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install and add to PATH
   - Install Russian and English language packs during installation

3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Run the setup verification:
   ```cmd
   verify_setup.bat
   ```

5. Start the application:
   ```cmd
   run_app.bat
   ```
   Or using PowerShell:
   ```powershell
   .\run_app.ps1
   ```

### macOS

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-assistant
   ```

2. Install Python 3 and Tesseract OCR (if not already installed):
   ```bash
   brew install python3 tesseract tesseract-lang poppler
   ```

3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Run the setup verification:
   ```bash
   ./verify_setup.sh
   ```

5. Start the application:
   ```bash
   ./run_app.sh
   ```

### Linux

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-assistant
   ```

2. Install Python 3 and Tesseract OCR (if not already installed):
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng poppler-utils
   ```

3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Run the setup verification:
   ```bash
   chmod +x verify_setup.sh
   ./verify_setup.sh
   ```

5. Start the application:
   ```bash
   chmod +x run_app.sh
   ./run_app.sh
   ```

## Usage

1. The application will start on `http://localhost:8501`
2. PDF documents are automatically processed from the `downloaded_pdfs` directory
3. **OCR Processing**: Scanned PDF pages without text are automatically processed with OCR (if Tesseract is installed)
4. Ask questions about your documents
5. The system will use RAG with knowledge graphs to provide accurate answers

### OCR Features

- **Automatic Detection**: Pages without extractable text are automatically processed with OCR
- **Multi-language Support**: Russian and English text recognition
- **High Quality**: 300 DPI resolution for better accuracy
- **Graceful Fallback**: If OCR is unavailable, scanned pages are skipped with a warning

## Project Structure

```
rag-assistant/
├── app.py                 # Streamlit web application
├── rag_assistant/        # Core RAG implementation
│   ├── config.py         # Configuration settings
│   ├── kg_retriever.py   # Knowledge Graph retriever
│   ├── knowledge_graph.py # Knowledge graph implementation
│   ├── main.py           # Main application logic
│   ├── self_rag.py       # Self-RAG implementation
│   ├── self_rag_langgraph.py # LangGraph integration
│   └── utils.py          # Utility functions
├── requirements.txt      # Python dependencies
├── Procfile             # Railway deployment configuration
├── run_app.bat          # Windows batch launcher
├── run_app.ps1          # Windows PowerShell launcher
├── run_app.sh           # macOS/Linux shell launcher
├── verify_setup.bat     # Windows setup verification
├── verify_setup.sh      # macOS/Linux setup verification
└── README.md            # This file
```

## Deployment

### Railway

The application is configured for Railway deployment with the `Procfile`. Simply connect your repository to Railway and deploy.

## Troubleshooting

### Common Issues

1. **Python not found**: Make sure Python 3.8+ is installed and in your PATH
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **API key issues**: Ensure your `.env` file contains a valid OpenAI API key
4. **Port conflicts**: The application uses port 8501 by default

### Platform-Specific Issues

- **Windows**: Make sure to run scripts as Administrator if needed
- **macOS**: Use `python3` and `pip3` commands
- **Linux**: Ensure proper permissions with `chmod +x` for shell scripts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on your platform
5. Submit a pull request

## License

This project is licensed under the MIT License.