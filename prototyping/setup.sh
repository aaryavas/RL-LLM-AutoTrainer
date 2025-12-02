#!/bin/bash
# RL-LLM-AutoTrainer Setup Script (Unix/macOS/Linux)
# Run with: chmod +x setup.sh && ./setup.sh

set -e

echo "========================================"
echo " RL-LLM-AutoTrainer Setup"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check Python
echo "[1/4] Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "  ERROR: Python not found. Please install Python 3.9+"
    exit 1
fi
echo "  Found: $($PYTHON_CMD --version)"

# Check Node.js
echo "[2/4] Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo "  ERROR: Node.js not found. Please install Node.js LTS"
    exit 1
fi
echo "  Found: Node.js $(node --version)"

# Install Python dependencies
echo "[3/4] Installing Python dependencies..."
echo "  This may take several minutes for PyTorch..."
pip install -r "$SCRIPT_DIR/requirements.txt"
echo "  Python dependencies installed successfully"

# Install Node.js dependencies and build CLI
echo "[4/4] Installing CLI wrapper dependencies..."
cd "$SCRIPT_DIR/cli-wrapper"
npm install
npm run build
echo "  CLI wrapper built successfully"

echo ""
echo "========================================"
echo " Setup Complete!"
echo "========================================"
echo ""
echo "Usage:"
echo "  cd cli-wrapper && npm start"
echo ""
echo "Or run directly:"
echo "  node prototyping/cli-wrapper/dist/index.js"
echo ""
