# RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) assistant with Knowledge Graph integration and Self-RAG capabilities.

## Features

- **Knowledge Graph RAG**: Enhanced retrieval using knowledge graphs
- **Self-RAG**: Self-reflective retrieval augmented generation with LangGraph
- **Document Processing**: PDF document ingestion and processing
- **Vector Database**: Efficient document storage and retrieval
- **Web Interface**: Streamlit-based user interface

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git

## Installation

### Windows

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-assistant
   ```

2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Run the setup verification:
   ```cmd
   verify_setup.bat
   ```

4. Start the application:
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

2. Install Python 3 (if not already installed):
   ```bash
   brew install python3
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

2. Install Python 3 (if not already installed):
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
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
2. Upload PDF documents through the web interface
3. Ask questions about your documents
4. The system will use RAG with knowledge graphs to provide accurate answers

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