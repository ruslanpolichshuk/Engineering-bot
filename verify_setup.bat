@echo off
echo ========================================
echo   RAG Assistant - Setup Verification
echo ========================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    goto :end
) else (
    echo [OK] Python is installed
)

echo.
echo Checking pip...
pip --version
if %errorlevel% neq 0 (
    echo [ERROR] pip not found!
    goto :end
) else (
    echo [OK] pip is available
)

echo.
echo Checking .env file...
if exist ".env" (
    echo [OK] .env file exists
) else (
    echo [WARNING] .env file not found - please create it from .env.template
)

echo.
echo Checking directories...
if exist "downloaded_pdfs" (
    echo [OK] downloaded_pdfs directory exists
) else (
    echo [INFO] downloaded_pdfs directory will be created on first run
)

if exist "vectordb" (
    echo [OK] vectordb directory exists
) else (
    echo [INFO] vectordb directory will be created on first run
)

echo.
echo Checking requirements.txt...
if exist "requirements.txt" (
    echo [OK] requirements.txt found
) else (
    echo [ERROR] requirements.txt not found!
    goto :end
)

echo.
echo ========================================
echo   Setup verification complete!
echo ========================================
echo.
echo Next steps:
echo 1. Create .env file with your OpenAI API key
echo 2. Run run_app.bat to start the application
echo.

:end
pause