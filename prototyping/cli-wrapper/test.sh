#!/bin/bash

# Comprehensive test script for CLI wrapper and Python finetuning module

echo "🧪 COMPREHENSIVE CLI TEST SUITE"
echo "================================"
echo ""

cd "$(dirname "$0")"
PROTOTYPING_DIR=$(dirname "$(pwd)")

# ============================================
# SECTION 1: TypeScript CLI Tests
# ============================================
echo "📦 SECTION 1: TypeScript CLI Tests"
echo "-----------------------------------"

echo ""
echo "✅ Test 1.1: Check if build succeeded"
if [ -f "dist/index.js" ]; then
    echo "   ✓ dist/index.js exists"
else
    echo "   ✗ dist/index.js not found"
    exit 1
fi

echo ""
echo "✅ Test 1.2: Check CLI help command"
node dist/index.js --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Help command works"
else
    echo "   ✗ Help command failed"
    exit 1
fi

echo ""
echo "✅ Test 1.3: Check CLI version command"
VERSION=$(node dist/index.js --version)
if [ "$VERSION" = "1.1.0" ]; then
    echo "   ✓ Version is 1.1.0"
else
    echo "   ✗ Version check failed (got: $VERSION)"
    exit 1
fi

echo ""
echo "✅ Test 1.4: Check dependencies"
npm list chalk boxen ora commander @inquirer/prompts > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ All npm dependencies installed"
else
    echo "   ⚠ Some dependencies may be missing"
fi

echo ""
echo "✅ Test 1.5: TypeScript compilation check"
npm run build > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ TypeScript compiles without errors"
else
    echo "   ✗ TypeScript compilation failed"
    exit 1
fi

echo ""
echo "✅ Test 1.6: Check config defaults alignment"
TS_EPOCHS=$(grep -A5 "DEFAULT_TRAINING" src/config/defaults.ts | grep "epochs" | grep -oP '\d+' | head -1)
TS_LR=$(grep -A5 "DEFAULT_TRAINING" src/config/defaults.ts | grep "learningRate" | grep -oP '[0-9]+e-[0-9]+' | head -1)
TS_LORA_R=$(grep -A5 "DEFAULT_VBLORA" src/config/defaults.ts | grep "loraR" | grep -oP '\d+' | head -1)

if [ "$TS_EPOCHS" = "5" ] && [ "$TS_LORA_R" = "16" ]; then
    echo "   ✓ TypeScript defaults aligned with Python CLI"
    echo "     epochs=$TS_EPOCHS, loraR=$TS_LORA_R"
else
    echo "   ⚠ TypeScript defaults may differ"
    echo "     epochs=$TS_EPOCHS (expected 5), loraR=$TS_LORA_R (expected 16)"
fi

# ============================================
# SECTION 2: Python Script Existence Tests
# ============================================
echo ""
echo "📦 SECTION 2: Python Script Existence Tests"
echo "--------------------------------------------"

echo ""
echo "✅ Test 2.1: Check data-gen.py exists"
if [ -f "$PROTOTYPING_DIR/data-gen.py" ]; then
    echo "   ✓ data-gen.py found"
else
    echo "   ✗ data-gen.py not found"
    exit 1
fi

echo ""
echo "✅ Test 2.2: Check VB-LoRA CLI exists"
if [ -f "$PROTOTYPING_DIR/finetuning/finetuners/cli.py" ]; then
    echo "   ✓ VB-LoRA cli.py found"
else
    echo "   ✗ VB-LoRA cli.py not found"
    exit 1
fi

echo ""
echo "✅ Test 2.3: Check ORPO CLI exists"
if [ -f "$PROTOTYPING_DIR/finetuning/finetuners/orpo/cli.py" ]; then
    echo "   ✓ ORPO cli.py found"
else
    echo "   ✗ ORPO cli.py not found"
    exit 1
fi

echo ""
echo "✅ Test 2.4: Check test suite exists"
if [ -f "$PROTOTYPING_DIR/finetuning/finetuners/tests/run_all_tests.py" ]; then
    echo "   ✓ Test suite found"
else
    echo "   ✗ Test suite not found"
    exit 1
fi

# ============================================
# SECTION 3: VB-LoRA CLI Tests
# ============================================
echo ""
echo "📦 SECTION 3: VB-LoRA CLI Tests"
echo "--------------------------------"

echo ""
echo "✅ Test 3.1: VB-LoRA CLI help command"
python "$PROTOTYPING_DIR/finetuning/finetuners/cli.py" --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ VB-LoRA CLI help works"
else
    echo "   ✗ VB-LoRA CLI help failed"
    exit 1
fi

echo ""
echo "✅ Test 3.2: VB-LoRA split subcommand"
python "$PROTOTYPING_DIR/finetuning/finetuners/cli.py" split --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ VB-LoRA split subcommand works"
else
    echo "   ✗ VB-LoRA split subcommand failed"
    exit 1
fi

echo ""
echo "✅ Test 3.3: VB-LoRA finetune subcommand"
python "$PROTOTYPING_DIR/finetuning/finetuners/cli.py" finetune --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ VB-LoRA finetune subcommand works"
else
    echo "   ✗ VB-LoRA finetune subcommand failed"
    exit 1
fi

echo ""
echo "✅ Test 3.4: VB-LoRA ORPO options present"
python "$PROTOTYPING_DIR/finetuning/finetuners/cli.py" finetune --help 2>&1 | grep -q "\-\-orpo"
if [ $? -eq 0 ]; then
    echo "   ✓ ORPO options available in VB-LoRA CLI"
else
    echo "   ✗ ORPO options not found"
    exit 1
fi

echo ""
echo "✅ Test 3.5: VB-LoRA model variants present"
python "$PROTOTYPING_DIR/finetuning/finetuners/cli.py" finetune --help 2>&1 | grep -q "SmolLM2-135M"
if [ $? -eq 0 ]; then
    echo "   ✓ Model variants available"
else
    echo "   ✗ Model variants not found"
    exit 1
fi

echo ""
echo "✅ Test 3.6: VB-LoRA preset options present"
python "$PROTOTYPING_DIR/finetuning/finetuners/cli.py" finetune --help 2>&1 | grep -q "quick_test"
if [ $? -eq 0 ]; then
    echo "   ✓ Preset options available"
else
    echo "   ✗ Preset options not found"
    exit 1
fi

# ============================================
# SECTION 4: ORPO CLI Tests
# ============================================
echo ""
echo "📦 SECTION 4: ORPO CLI Tests"
echo "-----------------------------"

echo ""
echo "✅ Test 4.1: ORPO CLI help command"
python "$PROTOTYPING_DIR/finetuning/finetuners/orpo/cli.py" --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ ORPO CLI help works"
else
    echo "   ✗ ORPO CLI help failed"
    exit 1
fi

echo ""
echo "✅ Test 4.2: ORPO beta parameter present"
python "$PROTOTYPING_DIR/finetuning/finetuners/orpo/cli.py" --help 2>&1 | grep -q "\-\-beta"
if [ $? -eq 0 ]; then
    echo "   ✓ ORPO beta parameter available"
else
    echo "   ✗ ORPO beta parameter not found"
    exit 1
fi

echo ""
echo "✅ Test 4.3: ORPO dry-run option present"
python "$PROTOTYPING_DIR/finetuning/finetuners/orpo/cli.py" --help 2>&1 | grep -q "\-\-dry-run"
if [ $? -eq 0 ]; then
    echo "   ✓ ORPO dry-run option available"
else
    echo "   ✗ ORPO dry-run option not found"
    exit 1
fi

# ============================================
# SECTION 5: Python Unit Tests
# ============================================
echo ""
echo "📦 SECTION 5: Python Unit Tests"
echo "--------------------------------"

echo ""
echo "✅ Test 5.1: Run ORPO generator tests"
python "$PROTOTYPING_DIR/finetuning/finetuners/tests/test_orpo_generator.py" > /tmp/orpo_test_output.txt 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ ORPO generator tests passed"
else
    echo "   ✗ ORPO generator tests failed"
    cat /tmp/orpo_test_output.txt
    exit 1
fi

echo ""
echo "✅ Test 5.2: Run comprehensive CLI tests"
python "$PROTOTYPING_DIR/finetuning/finetuners/tests/test_cli_comprehensive.py" > /tmp/cli_test_output.txt 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Comprehensive CLI tests passed"
else
    echo "   ✗ Comprehensive CLI tests failed"
    tail -50 /tmp/cli_test_output.txt
    exit 1
fi

echo ""
echo "✅ Test 5.3: Run integration tests"
python "$PROTOTYPING_DIR/finetuning/finetuners/tests/test_integration.py" > /tmp/integration_test_output.txt 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Integration tests passed"
else
    echo "   ✗ Integration tests failed"
    tail -50 /tmp/integration_test_output.txt
    exit 1
fi

# ============================================
# SECTION 6: Dry Run Tests
# ============================================
echo ""
echo "📦 SECTION 6: Dry Run Tests"
echo "----------------------------"

# Create temp test data
TEMP_DATA="/tmp/test_data_$$.csv"
echo "text,label" > "$TEMP_DATA"
for i in $(seq 1 20); do
    echo "test text $i,label_$((i % 3))" >> "$TEMP_DATA"
done

echo ""
echo "✅ Test 6.1: VB-LoRA dry run"
python "$PROTOTYPING_DIR/finetuning/finetuners/cli.py" finetune "$TEMP_DATA" --dry-run --variant SmolLM2-135M > /tmp/vblora_dryrun.txt 2>&1
if [ $? -eq 0 ] && grep -q "Dry run" /tmp/vblora_dryrun.txt; then
    echo "   ✓ VB-LoRA dry run works"
else
    echo "   ✗ VB-LoRA dry run failed"
    cat /tmp/vblora_dryrun.txt
    rm -f "$TEMP_DATA"
    exit 1
fi

# Create temp ORPO data
TEMP_ORPO_DATA="/tmp/test_orpo_data_$$.csv"
echo "prompt,chosen,rejected" > "$TEMP_ORPO_DATA"
for i in $(seq 1 10); do
    echo "question $i,correct answer $i,wrong answer $i" >> "$TEMP_ORPO_DATA"
done

echo ""
echo "✅ Test 6.2: ORPO dry run"
python "$PROTOTYPING_DIR/finetuning/finetuners/orpo/cli.py" --data_path "$TEMP_ORPO_DATA" --dry-run > /tmp/orpo_dryrun.txt 2>&1
if [ $? -eq 0 ] && grep -q "Dry run" /tmp/orpo_dryrun.txt; then
    echo "   ✓ ORPO dry run works"
else
    echo "   ✗ ORPO dry run failed"
    cat /tmp/orpo_dryrun.txt
    rm -f "$TEMP_DATA" "$TEMP_ORPO_DATA"
    exit 1
fi

# Cleanup temp files
rm -f "$TEMP_DATA" "$TEMP_ORPO_DATA"

# ============================================
# SECTION 7: Data Split Test
# ============================================
echo ""
echo "📦 SECTION 7: Data Split Test"
echo "------------------------------"

TEMP_SPLIT_DATA="/tmp/test_split_data_$$.csv"
TEMP_SPLIT_DIR="/tmp/test_split_output_$$"

echo "text,label" > "$TEMP_SPLIT_DATA"
for i in $(seq 1 100); do
    echo "sample text $i,label_$((i % 5))" >> "$TEMP_SPLIT_DATA"
done

echo ""
echo "✅ Test 7.1: Data split command"
python "$PROTOTYPING_DIR/finetuning/finetuners/cli.py" split "$TEMP_SPLIT_DATA" --output-dir "$TEMP_SPLIT_DIR" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    if [ -f "$TEMP_SPLIT_DIR/train.csv" ] && [ -f "$TEMP_SPLIT_DIR/val.csv" ] && [ -f "$TEMP_SPLIT_DIR/test.csv" ]; then
        echo "   ✓ Data split creates all files"
        
        TRAIN_LINES=$(wc -l < "$TEMP_SPLIT_DIR/train.csv")
        VAL_LINES=$(wc -l < "$TEMP_SPLIT_DIR/val.csv")
        TEST_LINES=$(wc -l < "$TEMP_SPLIT_DIR/test.csv")
        echo "   ✓ Split sizes: train=$TRAIN_LINES, val=$VAL_LINES, test=$TEST_LINES"
    else
        echo "   ✗ Some split files missing"
        exit 1
    fi
else
    echo "   ✗ Data split command failed"
    exit 1
fi

# Cleanup
rm -rf "$TEMP_SPLIT_DATA" "$TEMP_SPLIT_DIR"

# ============================================
# FINAL SUMMARY
# ============================================
echo ""
echo "================================"
echo "✅ ALL TESTS PASSED!"
echo "================================"
echo ""
echo "Summary:"
echo "  - TypeScript CLI: OK"
echo "  - Python Scripts: OK"
echo "  - VB-LoRA CLI: OK"
echo "  - ORPO CLI: OK"
echo "  - Unit Tests: OK"
echo "  - Dry Runs: OK"
echo "  - Data Split: OK"
echo ""
echo "To run the CLI:"
echo "  npm start"
echo ""
echo "Or from anywhere (if globally linked):"
echo "  synth-data"
