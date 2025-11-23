#!/bin/bash
set -e

echo "📦 Installing dependencies..."
pip install bert_score rouge_score absl-py nltk

echo "🚀 Running test_dpo_metrics.py..."
python3 test_dpo_metrics.py
