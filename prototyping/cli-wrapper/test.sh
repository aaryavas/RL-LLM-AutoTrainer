#!/bin/bash

# Test script for TypeScript CLI

echo "🧪 Testing TypeScript CLI Tool"
echo "================================"
echo ""

cd "$(dirname "$0")"

echo "✅ Test 1: Check if build succeeded"
if [ -f "dist/index.js" ]; then
    echo "   ✓ dist/index.js exists"
else
    echo "   ✗ dist/index.js not found"
    exit 1
fi

echo ""
echo "✅ Test 2: Check CLI help command"
node dist/index.js --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Help command works"
else
    echo "   ✗ Help command failed"
    exit 1
fi

echo ""
echo "✅ Test 3: Check CLI version command"
VERSION=$(node dist/index.js --version)
if [ "$VERSION" = "1.0.0" ]; then
    echo "   ✓ Version is 1.0.0"
else
    echo "   ✗ Version check failed (got: $VERSION)"
    exit 1
fi

echo ""
echo "✅ Test 4: Check if Python script exists"
if [ -f "../data-gen.py" ]; then
    echo "   ✓ data-gen.py found at expected location"
else
    echo "   ✗ data-gen.py not found at ../data-gen.py"
    exit 1
fi

echo ""
echo "✅ Test 5: Check dependencies"
npm list chalk boxen ora commander @inquirer/prompts > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ All npm dependencies installed"
else
    echo "   ⚠ Some dependencies may be missing"
fi

echo ""
echo "✅ Test 6: TypeScript compilation check"
npm run build > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ TypeScript compiles without errors"
else
    echo "   ✗ TypeScript compilation failed"
    exit 1
fi

echo ""
echo "================================"
echo "✅ All tests passed!"
echo ""
echo "To run the CLI:"
echo "  npm start"
echo ""
echo "Or from anywhere (if globally linked):"
echo "  synth-data"
