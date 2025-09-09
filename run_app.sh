#!/bin/bash

echo "========================================"
echo "   RAG Assistant - macOS Launcher"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ using Homebrew: brew install python3"
    echo "Or download from https://python.org"
    exit 1
fi

# Check Python version
python3 --version

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found!"
    echo "Please create .env file with your OpenAI API key"
    echo "See .env.template for reference"
    echo
fi

# Create necessary directories
mkdir -p downloaded_pdfs
mkdir -p vectordb

# Install/update dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Run the application
echo
echo "Starting RAG Assistant..."
echo "The application will open in your default web browser"
echo "Press Ctrl+C to stop the application"
echo
streamlit run app.py --server.port=8501 --server.address=localhost