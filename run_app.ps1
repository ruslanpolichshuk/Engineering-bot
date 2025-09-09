# RAG Assistant - PowerShell Launcher
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   RAG Assistant - Windows Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "WARNING: .env file not found!" -ForegroundColor Yellow
    Write-Host "Please create .env file with your OpenAI API key" -ForegroundColor Yellow
    Write-Host "See .env.template for reference" -ForegroundColor Yellow
    Write-Host ""
}

# Create necessary directories
if (-not (Test-Path "downloaded_pdfs")) {
    New-Item -ItemType Directory -Name "downloaded_pdfs" | Out-Null
    Write-Host "Created downloaded_pdfs directory" -ForegroundColor Green
}

if (-not (Test-Path "vectordb")) {
    New-Item -ItemType Directory -Name "vectordb" | Out-Null
    Write-Host "Created vectordb directory" -ForegroundColor Green
}

# Install/update dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Run the application
Write-Host ""
Write-Host "Starting RAG Assistant..." -ForegroundColor Green
Write-Host "The application will open in your default web browser" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Cyan
Write-Host ""
streamlit run app.py --server.port=8501 --server.address=localhost

Read-Host "Press Enter to exit"