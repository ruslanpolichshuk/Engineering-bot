@echo off
echo ========================================
echo   RAG Assistant - Windows Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo WARNING: .env file not found!
    echo Please create .env file with your OpenAI API key
    echo See .env.template for reference
    echo.
)

REM Create necessary directories
if not exist "downloaded_pdfs" mkdir downloaded_pdfs
if not exist "vectordb" mkdir vectordb

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run the application
echo.
echo Starting RAG Assistant...
echo The application will open in your default web browser
echo Press Ctrl+C to stop the application
echo.
streamlit run app.py --server.port=8501 --server.address=localhost

pause