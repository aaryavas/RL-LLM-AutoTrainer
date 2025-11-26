# ORPO Fine-tuning Module

This module implements **Odds Ratio Preference Optimization (ORPO)** for fine-tuning Language Models (LLMs) on preference data. ORPO is a reference-model-free preference optimization algorithm that combines Supervised Fine-Tuning (SFT) and preference alignment into a single process.

## Features

- **Reference-Free**: Does not require a separate reference model, saving memory and compute.
- **Integrated Pipeline**: Handles data loading, splitting, training, and visualization.
- **Visualization**: Automatically generates plots for rewards, margins, and accuracies.
- **Configurable**: Supports various training parameters via CLI or Python API.

## Prerequisites

Ensure you have the project dependencies installed. This module relies on `trl`, `transformers`, `datasets`, `peft`, and `pandas`.

## Data Format

The module expects a CSV file containing preference data. The CSV should have at least three columns:

1.  **Prompt**: The input instruction or query.
2.  **Chosen**: The preferred response.
3.  **Rejected**: The disfavored response.

**Example `preference_data.csv`:**

```csv
prompt,chosen,rejected
"What is the capital of France?","The capital of France is Paris.","Paris is the capital."
"Write a python function to add two numbers.","def add(a, b):\n    return a + b","function add(a, b) { return a + b; }"
```

## Usage

### 1. Command Line Interface (CLI)

You can run the training directly from the command line using `finetuning/orpo/cli.py`.

**Basic Usage:**

```bash
python finetuning/orpo/cli.py \
    --model_name Qwen/Qwen2-0.5B-Instruct \
    --data_path path/to/your/preference_data.csv \
    --output_dir ./output/my_orpo_model
```

**Advanced Usage:**

```bash
python finetuning/orpo/cli.py \
    --model_name Qwen/Qwen2-0.5B-Instruct \
    --data_path data/preferences.csv \
    --output_dir ./output/qwen_orpo \
    --prompt_column question \
    --chosen_column good_answer \
    --rejected_column bad_answer \
    --epochs 3 \
    --batch_size 2 \
    --lr 5e-6 \
    --beta 0.1 \
    --max_prompt_length 1024
```

**Arguments:**

- `--model_name`: Base model ID (Hugging Face) or path. Default: `Qwen/Qwen2-0.5B-Instruct`.
- `--data_path`: Path to the CSV file containing preference data. (Required)
- `--output_dir`: Directory to save the model and logs. Default: `./output/orpo_models`.
- `--prompt_column`: Column name for prompts. Default: `prompt`.
- `--chosen_column`: Column name for chosen responses. Default: `chosen`.
- `--rejected_column`: Column name for rejected responses. Default: `rejected`.
- `--epochs`: Number of training epochs. Default: `3`.
- `--batch_size`: Training batch size per device. Default: `4`.
- `--lr`: Learning rate. Default: `1e-5`.
- `--beta`: ORPO beta (lambda) parameter. Default: `0.1`.
- `--max_prompt_length`: Maximum length for prompts. Default: `512`.

### 2. Python API

You can also use the `ORPOFineTuner` class in your own scripts.

```python
from finetuning.orpo.train_orpo import ORPOFineTuner

# Initialize
tuner = ORPOFineTuner(
    model_name="Qwen/Qwen2-0.5B-Instruct",
    output_dir="./output/my_custom_run"
)

# Run Pipeline
tuner.finetune_pipeline(
    data_path="data/my_preferences.csv",
    num_train_epochs=1,
    batch_size=4,
    beta=0.1
)
```

## Outputs

After training, the `output_dir` will contain:

- **Model Checkpoints**: The fine-tuned model weights.
- **`orpo_metadata.json`**: Configuration and final metrics.
- **Plots**:
    - `rewards_plot.png`: Chosen vs. Rejected rewards over time.
    - `margins_plot.png`: The margin between chosen and rejected rewards.
    - `accuracy_plot.png`: Classification accuracy of the preference task.
- **Logs**: Training logs.

## Directory Structure

```
finetuning/orpo/
├── cli.py                  # CLI Entry point
├── train_orpo.py           # Main Orchestrator
├── config/                 # Configuration classes
├── core/                   # Data processing
├── training/               # Trainer wrapper and visualization
└── utils/                  # Helpers
```
