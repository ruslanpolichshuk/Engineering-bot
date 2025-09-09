#!/bin/bash

echo "========================================"
echo "   RAG Assistant - Setup Verification"
echo "========================================"
echo

echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    python3 --version
    echo "[OK] Python 3 is installed"
else
    echo "[ERROR] Python 3 not found!"
    echo "Please install Python 3.8+ using Homebrew: brew install python3"
    exit 1
fi

echo
echo "Checking pip..."
if command -v pip3 &> /dev/null; then
    pip3 --version
    echo "[OK] pip3 is available"
else
    echo "[ERROR] pip3 not found!"
    echo "Please install pip3: python3 -m ensurepip --upgrade"
    exit 1
fi

echo
echo "Checking .env file..."
if [ -f ".env" ]; then
    echo "[OK] .env file exists"
else
    echo "[WARNING] .env file not found - please create it from .env.template"
fi

echo
echo "Checking directories..."
if [ -d "downloaded_pdfs" ]; then
    echo "[OK] downloaded_pdfs directory exists"
else
    echo "[INFO] downloaded_pdfs directory will be created on first run"
fi

if [ -d "vectordb" ]; then
    echo "[OK] vectordb directory exists"
else
    echo "[INFO] vectordb directory will be created on first run"
fi

echo
echo "Checking requirements.txt..."
if [ -f "requirements.txt" ]; then
    echo "[OK] requirements.txt found"
else
    echo "[ERROR] requirements.txt not found!"
    exit 1
fi

echo
echo "========================================"
echo "   Setup verification complete!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Create .env file with your OpenAI API key"
echo "2. Run ./run_app.sh to start the application"
echo "3. Make sure to chmod +x run_app.sh if needed"
echo