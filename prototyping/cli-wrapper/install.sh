#!/bin/bash

# Synthetic Data Generator TypeScript CLI - Installation Script

echo "🚀 Installing Synthetic Data Generator TypeScript CLI..."
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    echo "Visit: https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js found: $(node --version)"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ npm found: $(npm --version)"

# Navigate to cli-wrapper directory
cd "$(dirname "$0")"

# Install dependencies
echo ""
echo "📦 Installing npm dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed"

# Build TypeScript
echo ""
echo "🔨 Building TypeScript..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo "✅ Build completed"

# Ask if user wants to link globally
echo ""
read -p "Do you want to install the CLI globally? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    npm link
    if [ $? -eq 0 ]; then
        echo "✅ CLI installed globally as 'synth-data'"
        echo ""
        echo "You can now run: synth-data"
    else
        echo "❌ Global installation failed"
    fi
else
    echo "ℹ️  Skipped global installation"
    echo ""
    echo "You can run the CLI with: npm start"
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Quick start:"
echo "  npm start        # Run the CLI"
echo "  npm run dev      # Run in development mode"
echo "  synth-data       # Run globally (if linked)"
