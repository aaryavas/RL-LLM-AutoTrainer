# SmolLM2 Fine-Tuning with VB-LoRA

This module provides a complete pipeline for fine-tuning SmolLM2 models using **VB-LoRA** (Vector Bank Low-Rank Adaptation) PEFT on text data.

## What is VB-LoRA?

VB-LoRA extends the standard LoRA approach by introducing a **vector bank** - a shared pool of learnable vectors that can be combined to create adapter weights. This provides:

- **Higher parameter efficiency**: Better performance with fewer trainable parameters
- **Flexible capacity**: Adjustable vector bank size for different task complexities
- **Memory efficiency**: Compatible with 4-bit/8-bit quantization
- **Separate learning rates**: Independent optimization for vector bank, logits, and base parameters

## Features

- 🤖 **Support for all SmolLM2 variants** (135M, 360M, 1.7B parameters)
- 🔧 **VB-LoRA PEFT integration** for efficient fine-tuning
- 📊 **Automatic data splitting** (train/validation/test)
- ⚙️ **Configurable training parameters** with presets
- 💾 **4-bit/8-bit quantization** for memory efficiency
- 🖥️ **CLI and Python API** interfaces
- 📈 **Comprehensive training callbacks** and metrics

## Installation

1. **Clone or navigate to the directory**:
```bash
cd finetuning-vblora
```

2. **Run the setup script**:
```bash
bash setup.sh
```

3. **Set up your Hugging Face token** in `.env` file:
```bash
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

## Quick Start

### Using the CLI

1. **Split your data** (optional):
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

5. **Advanced fine-tuning with VB-LoRA parameters**:
```bash
python cli.py finetune data.csv \\
    --epochs 5 \\
    --lr 1e-4 \\
    --batch-size 4 \\
    --num-vectors 2048 \\
    --lora-r 8 \\
    --early-stopping 3
```

### Using Python API

```python
from finetuning import SmolLM2VBLoRAFineTuner

# Initialize fine-tuner
finetuner = SmolLM2VBLoRAFineTuner(
    model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    output_dir="./my_models"
)

# Configure VB-LoRA
finetuner.vblora_config.num_vectors = 128
finetuner.vblora_config.lora_r = 8

# Run complete pipeline
model_path, eval_results = finetuner.finetune_pipeline(
    data_path="my_data.csv",
    num_train_epochs=3,
    learning_rate=2e-4,
    batch_size=4
)

print(f"Model saved to: {model_path}")
```

## Data Format

Your data CSV should have the following columns:
- `text`: The input text
- `label`: The target label/output

Example:
```csv
text,label
"Classify this text","Positive"
"Another example","Negative"
```

## VB-LoRA Configuration

### Key Parameters

- **`num_vectors`**: Number of vectors in the vector bank (default: 90)
  - Larger values = higher capacity but more memory
  - Recommended: 64-128 for small models, 256-2048 for larger models

- **`lora_r`**: LoRA rank (default: 4)
  - Controls adapter dimensionality
  - Recommended: 4-8 for most tasks

- **`learning_rate_vector_bank`**: Learning rate for vector bank (default: 1e-3)
- **`learning_rate_logits`**: Learning rate for logits (default: 1e-2)
- **`learning_rate`**: Base learning rate for other parameters (default: 2e-4)

### VB-LoRA Presets

```bash
# Quick test (32 vectors)
python cli.py finetune data.csv --num-vectors 32 --lora-r 4

# Standard (90 vectors)
python cli.py finetune data.csv --preset standard

# High capacity (2048 vectors)
python cli.py finetune data.csv --num-vectors 2048 --lora-r 8
```

## Model Variants

| Variant | Parameters | Recommended Vectors | Batch Size | Learning Rate |
|---------|------------|---------------------|------------|---------------|
| SmolLM2-135M | 135M | 64 | 16 | 5e-4 |
| SmolLM2-360M | 360M | 90 | 12 | 3e-4 |
| SmolLM2-1.7B | 1.7B | 128 | 4 | 2e-4 |

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
- `--preset`: Use configuration preset (quick_test, standard, thorough, memory_efficient)
- `--epochs`: Number of training epochs
- `--lr`: Base learning rate
- `--batch-size`: Batch size per device
- `--num-vectors`: Number of vectors in VB-LoRA vector bank
- `--lora-r`: LoRA rank
- `--bits`: Quantization bits (4, 8, 16, 32)
- `--dry-run`: Show configuration without training

## Output Structure

After fine-tuning, you'll get:
```
output/vblora_models/
└── vblora_finetune_20241110_123456/
    ├── adapter_model/           # VB-LoRA adapter weights
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    ├── vblora_metadata.json     # Training configuration
    └── logs/                    # Training logs
```

## Memory Requirements

| Model | GPU Memory (4-bit) | GPU Memory (8-bit) |
|-------|-------------------|-------------------|
| SmolLM2-135M | ~2 GB | ~3 GB |
| SmolLM2-360M | ~3 GB | ~5 GB |
| SmolLM2-1.7B | ~6 GB | ~10 GB |

For limited GPU memory:
- Use 4-bit quantization: `--bits 4` (default)
- Use smaller batch sizes: `--batch-size 1` or `--batch-size 2`
- Use memory-efficient preset: `--preset memory_efficient`
- Reduce vector bank size: `--num-vectors 64`

## Architecture

The codebase follows OOP-first design with strict file size limits:

```
finetuning-vblora/
├── config/              # Configuration classes (<150 lines each)
│   ├── base_config.py
│   ├── vblora_config.py
│   └── model_variants.py
├── core/                # Core components (<250 lines each)
│   ├── data_processor.py
│   ├── tokenizer_manager.py
│   ├── model_loader.py
│   └── optimizer_factory.py
├── training/            # Training orchestration (<250 lines each)
│   ├── trainer.py
│   ├── callbacks.py
│   └── metrics.py
├── utils/               # Utilities (<200 lines each)
│   ├── data_splitter.py
│   └── helpers.py
├── finetuning.py        # Main API (~300 lines)
└── cli.py               # CLI interface (~300 lines)
```

## Comparison with Standard LoRA

| Feature | Standard LoRA | VB-LoRA |
|---------|--------------|---------|
| Adapter structure | Direct low-rank matrices | Vector bank + composition |
| Parameter efficiency | Good | Better |
| Flexibility | Fixed capacity | Adjustable (num_vectors) |
| Learning rates | Single LR | Separate LRs for components |
| Memory | Efficient | More efficient |

## Examples

### Basic Classification Task
```bash
# Generate synthetic data (using your data generator)
python ../data-gen.py --sample_size 1000 --output_dir ./data

# Fine-tune with VB-LoRA
python cli.py finetune ./data/samples.csv --epochs 3 --num-vectors 90
```

### High-Capacity Fine-Tuning
```bash
# For complex tasks requiring more capacity
python cli.py finetune data.csv \\
    --variant SmolLM2-1.7B \\
    --num-vectors 2048 \\
    --lora-r 8 \\
    --epochs 5 \\
    --batch-size 2
```

### Memory-Constrained Environment
```bash
# Optimize for limited GPU memory
python cli.py finetune data.csv \\
    --preset memory_efficient \\
    --bits 4 \\
    --batch-size 1 \\
    --num-vectors 32
```

## Troubleshooting

### CUDA out of memory
- Reduce batch size: `--batch-size 1`
- Reduce vector bank: `--num-vectors 32`
- Use 4-bit quantization: `--bits 4`

### Model not found
- Check your Hugging Face token in `.env`
- Ensure internet connection
- Verify model name spelling

### Slow training
- Increase batch size if memory allows
- Use gradient accumulation
- Enable bfloat16: `--bf16`

## License

This project is part of the RL-LLM-Dupe repository. See the main repository for license information.

## Credits

Based on the VB-LoRA paper and implementation, adapted for SmolLM2 models with user-friendly CLI and Python API.
