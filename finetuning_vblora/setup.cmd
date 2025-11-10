@echo off
setlocal

echo 🚀 VB-LoRA Fine-tuning Setup
echo =============================

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Python is required but not installed. Please install Python and add it to your PATH.
    goto :eof
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ Python found: %PYTHON_VERSION%

REM Check if pip is available
where pip >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ pip is required but not installed.
    goto :eof
)
for /f "tokens=*" %%i in ('pip --version') do set PIP_VERSION=%%i
echo ✅ pip found: %PIP_VERSION%

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo ❌ Failed to create virtual environment.
        goto :eof
    )
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to activate virtual environment.
    echo    Please try running 'venv\Scripts\activate.bat' manually.
    goto :eof
)


REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to upgrade pip.
    goto :eof
)

REM Install requirements
echo 📦 Installing Python packages...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo ❌ Failed to install requirements from requirements.txt.
        goto :eof
    )
) else (
    echo ⚠️  requirements.txt not found, installing core packages...
    pip install torch transformers peft datasets accelerate bitsandbytes pandas numpy scikit-learn evaluate python-dotenv huggingface-hub
    if %ERRORLEVEL% neq 0 (
        echo ❌ Failed to install core packages.
        goto :eof
    )
)

REM Check CUDA availability
echo 🔍 Checking CUDA availability...
python -c "import torch; print(torch.__version__); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo 📝 Creating .env file...
    (
        echo # Add your Hugging Face token here
        echo HF_TOKEN=your_huggingface_token_here
    ) > .env
    echo ⚠️  Please edit .env file and add your Hugging Face token
)

REM Create output directories
echo 📁 Creating output directories...
if not exist "output\vblora_models" mkdir "output\vblora_models"
if not exist "split_data" mkdir "split_data"
if not exist "logs" mkdir "logs"

echo.
echo 🎉 Setup completed successfully!
echo.
echo Next steps:
echo 1. Edit .env file and add your Hugging Face token
echo 2. Run: python cli.py --help
echo 3. Try: python cli.py finetune your_data.csv --preset quick_test
echo.
echo To activate the environment later, run:
echo venv\Scripts\activate.bat

endlocal
