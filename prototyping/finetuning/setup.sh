#!/bin/bash

# SmolLM2 Fine-tuning Setup Script
# This script helps set up the environment for fine-tuning SmolLM2 models

set -e  # Exit on any error

echo "🚀 SmolLM2 Fine-tuning Setup"
echo "============================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed"
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing Python packages..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found, installing core packages..."
    pip install torch transformers peft datasets accelerate pandas numpy scikit-learn evaluate python-dotenv huggingface-hub
fi

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    echo "# Add your Hugging Face token here" > .env
    echo "HF_TOKEN=your_huggingface_token_here" >> .env
    echo "⚠️ Please edit .env file and add your Hugging Face token"
fi

# Create output directories
echo "📁 Creating output directories..."
mkdir -p finetuned_models
mkdir -p split_data
mkdir -p logs

# Test the installation
echo "🧪 Testing installation..."
python3 test.py

# Check if everything is working
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file and add your Hugging Face token"
    echo "2. Run: python cli.py --help"
    echo "3. Try: python cli.py finetune your_data.csv --preset quick_test"
    echo ""
    echo "To activate the environment later, run:"
    echo "source venv/bin/activate"
else
    echo ""
    echo "❌ Setup completed but tests failed"
    echo "Please check the error messages above"
fi