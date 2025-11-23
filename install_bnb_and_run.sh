#!/bin/bash
set -e

echo "📦 Installing bitsandbytes and accelerate..."
pip install bitsandbytes accelerate

echo "🚀 Running run_finetune_and_dpo_selection.py..."
python3 run_finetune_and_dpo_selection.py
