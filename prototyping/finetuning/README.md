# SmolLM2 Fine-Tuning with LoRA PEFT

This module provides a complete pipeline for fine-tuning SmolLM2 models using LoRA (Low-Rank Adaptation) PEFT on synthetic text classification data.

## Features

- 🤖 **Support for all SmolLM2 variants** (135M, 360M, 1.7B parameters)
- 🔧 **LoRA PEFT integration** for efficient fine-tuning
- 📊 **Automatic data splitting** (train/validation/test)
- ⚙️ **Configurable training parameters** with presets
- 📈 **Comprehensive evaluation metrics** (accuracy, precision, recall, F1)
- 💾 **Model checkpointing** and early stopping
- 🖥️ **CLI interface** for easy usage
- 📱 **Mixed precision training** for memory efficiency

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Hugging Face token in a `.env` file:
```bash
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

## Quick Start

### Using the CLI

1. **Split your synthetic data** (optional):
```bash
python cli.py split data.csv --output-dir ./split_data
```

2. **Fine-tune SmolLM2** with default settings:
```bash
python cli.py finetune data.csv
```

3. **Use a preset configuration**:
```bash
python cli.py finetune data.csv --preset standard
```

4. **Use a specific model variant**:
```bash
python cli.py finetune data.csv --variant SmolLM2-360M
```

5. **Advanced fine-tuning**:
```bash
python cli.py finetune data.csv --epochs 5 --lr 1e-4 --batch-size 4 --early-stopping 3
```

### Using Python API

```python
from finetuning import SmolLM2FineTuner

# Initialize fine-tuner
finetuner = SmolLM2FineTuner(
    model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    output_dir="./my_models"
)

# Run complete pipeline
model_path, eval_results = finetuner.finetune_pipeline(
    data_path="my_synthetic_data.csv",
    num_train_epochs=3,
    learning_rate=2e-4,
    batch_size=8
)

print(f"Model saved to: {model_path}")
print(f"Evaluation results: {eval_results}")
```

## Data Format

Your synthetic data CSV should have the following columns:
- `text`: The input text for classification
- `label`: The target label/class

Example:
```csv
text,label,model,reasoning
"This is a positive example",positive,HuggingFaceTB/SmolLM2-1.7B-Instruct,"The text expresses positive sentiment"
"This is a negative example",negative,HuggingFaceTB/SmolLM2-1.7B-Instruct,"The text expresses negative sentiment"
```

## Model Variants

| Variant | Parameters | Recommended Batch Size | Learning Rate |
|---------|------------|----------------------|---------------|
| SmolLM2-135M | 135M | 16 | 5e-4 |
| SmolLM2-360M | 360M | 12 | 3e-4 |
| SmolLM2-1.7B | 1.7B | 8 | 2e-4 |

## Configuration Presets

- **`quick_test`**: Fast training for testing (1 epoch, no evaluation)
- **`standard`**: Balanced training (3 epochs, standard settings)
- **`thorough`**: Comprehensive training (5 epochs, lower learning rate)
- **`memory_efficient`**: Optimized for limited GPU memory

## CLI Commands

### Split Data
```bash
python cli.py split <data_path> [options]
```

Options:
- `--output-dir`: Output directory (default: `./split_data`)
- `--test-size`: Test set proportion (default: 0.2)
- `--val-size`: Validation set proportion (default: 0.1)
- `--text-column`: Text column name (default: `text`)
- `--label-column`: Label column name (default: `label`)

### Fine-tune Model
```bash
python cli.py finetune <data_path> [options]
```

Key options:
- `--model`: Specific model name
- `--variant`: Use predefined variant (SmolLM2-135M, SmolLM2-360M, SmolLM2-1.7B)
- `--preset`: Use configuration preset
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--batch-size`: Batch size per device
- `--output-dir`: Output directory for models
- `--dry-run`: Show configuration without training

## LoRA Configuration

The module uses the following LoRA settings by default:
- **Rank (r)**: 16
- **Alpha**: 32
- **Dropout**: 0.1
- **Target modules**: All attention and MLP projection layers

You can customize these in `config.py`.

## Output Structure

After fine-tuning, you'll get:
```
finetuned_models/
└── smollm2_finetune_20241107_123456/
    ├── adapter_config.json       # LoRA configuration
    ├── adapter_model.safetensors # LoRA weights
    ├── label_mapping.json        # Label to ID mapping
    ├── training_metrics.json     # Training metrics
    ├── results_summary.json      # Complete results
    └── logs/                     # Training logs
```

## Memory Requirements

| Model | GPU Memory (FP16) | GPU Memory (FP32) |
|-------|------------------|------------------|
| SmolLM2-135M | ~2 GB | ~4 GB |
| SmolLM2-360M | ~4 GB | ~8 GB |
| SmolLM2-1.7B | ~8 GB | ~16 GB |

For limited GPU memory, use:
- Smaller batch sizes (`--batch-size 2` or `--batch-size 4`)
- Memory-efficient preset (`--preset memory_efficient`)
- CPU training (`--cpu-only`)

## Examples

### Basic Classification Task
```bash
# Generate synthetic data (using your data generator)
python ../data-gen.py --sample_size 1000 --output_dir ../generated_data

# Fine-tune on the generated data
python cli.py finetune ../generated_data/20241107_123456.csv --epochs 3
```

### Sentiment Analysis
```bash
# Fine-tune for sentiment analysis with specific settings
python cli.py finetune sentiment_data.csv \
    --variant SmolLM2-360M \
    --epochs 5 \
    --lr 1e-4 \
    --batch-size 8 \
    --early-stopping 3 \
    --run-name sentiment_classifier
```

### Memory-Constrained Environment
```bash
# Fine-tune with memory optimization
python cli.py finetune data.csv \
    --preset memory_efficient \
    --batch-size 2 \
    --no-fp16
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size: `--batch-size 2`
   - Use CPU: `--cpu-only`
   - Disable mixed precision: `--no-fp16`

2. **Model not found**:
   - Check your Hugging Face token
   - Ensure internet connection
   - Verify model name spelling

3. **Data format errors**:
   - Ensure CSV has `text` and `label` columns
   - Check for missing values
   - Verify text encoding (UTF-8)

### Performance Tips

1. **Faster training**:
   - Use smaller model variants (135M or 360M)
   - Increase batch size if memory allows
   - Use fewer epochs for initial testing

2. **Better results**:
   - Use more training data
   - Increase number of epochs
   - Try different learning rates
   - Use early stopping to prevent overfitting

## Integration with Data Generator

This fine-tuning module is designed to work seamlessly with the synthetic data generator:

```bash
# 1. Generate synthetic data
python ../data-gen.py --sample_size 5000 --output_dir ../generated_data

# 2. Fine-tune on generated data
python cli.py finetune ../generated_data/latest.csv --preset standard

# 3. Evaluate results
python cli.py finetune ../generated_data/latest.csv --dry-run  # Show config only
```

## Next Steps

After fine-tuning, you can:
1. Load the model using `PeftModel.from_pretrained()`
2. Use it for inference on new data
3. Further fine-tune on domain-specific data
4. Integrate into your application pipeline

For more advanced usage, see the Python API documentation in `finetuning.py`.