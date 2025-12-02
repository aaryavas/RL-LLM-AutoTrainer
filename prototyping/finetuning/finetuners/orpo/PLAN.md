# ORPO Training Implementation Plan

This document outlines the plan for implementing ORPO (Odds Ratio Preference Optimization) training in the `finetuning/orpo/` directory, following the structure of the existing VB-LoRA implementation.

## Directory Structure

```
finetuning/orpo/
├── __init__.py
├── cli.py                  # CLI entry point for ORPO training
├── train_orpo.py           # Main orchestrator (ORPOFineTuner class)
├── config/
│   ├── __init__.py
│   └── orpo_config.py      # ORPO-specific configuration
├── core/
│   ├── __init__.py
│   └── data_processor.py   # Data processor for preference data
├── training/
│   ├── __init__.py
│   ├── trainer.py          # ORPOTrainer wrapper
│   └── visualization.py    # Graph generation for rewards/margins
└── utils/
    ├── __init__.py
    └── helpers.py          # Helper functions
```

## Components

### 1. Configuration (`config/orpo_config.py`)
- **`ORPOSpecificConfig`**: Dataclass for ORPO-specific parameters.
    - `beta`: float (default 0.1) - The lambda parameter for ORPO loss.
    - `max_prompt_length`: int (default 512)
    - `max_completion_length`: int (default 1024)
    - `disable_dropout`: bool (default True)

### 2. Data Processing (`core/data_processor.py`)
- **`PreferenceDataProcessor`**:
    - Handles loading of preference datasets (chosen/rejected).
    - Supports CSV input with columns for prompt, chosen, rejected.
    - Formats data for `trl.ORPOTrainer`.
    - Uses `DataSplitter` from `finetuning/utils/data_splitter.py` (reused).

### 3. Training Orchestration (`train_orpo.py`)
- **`ORPOFineTuner`**:
    - Similar to `SmolLM2VBLoRAFineTuner`.
    - Initializes configs.
    - `finetune_pipeline()`:
        1. Load and split data.
        2. Setup tokenizer.
        3. Setup model (Standard AutoModelForCausalLM, potentially with PEFT/LoRA if needed, but ORPO is often full fine-tune or LoRA. We will support LoRA via PEFT config).
        4. Prepare datasets.
        5. Setup `ORPOTrainerWrapper`.
        6. Train.
        7. Generate visualization.
        8. Save model and metadata.

### 4. Trainer Wrapper (`training/trainer.py`)
- **`ORPOTrainerWrapper`**:
    - Wraps `trl.ORPOTrainer`.
    - Sets up `trl.ORPOConfig`.
    - Handles callbacks (including `EpochMetricsCallback` if applicable, though ORPO logs metrics differently).
    - `train()` method.

### 5. Visualization (`training/visualization.py`)
- **`TrainingVisualizer`**:
    - `plot_metrics(log_history, output_dir)`:
        - Extracts `log_odds_chosen`, `rewards/chosen`, `rewards/rejected`, `rewards/margins`, `rewards/accuracies`.
        - Plots these over epochs/steps.
        - Saves the plot as an image file in `output_dir`.

### 6. CLI (`cli.py`)
- Uses `argparse` to expose ORPO training functionality.

## Implementation Steps

1.  **Create Configs**: Define `ORPOSpecificConfig`.
2.  **Create Data Processor**: Implement `PreferenceDataProcessor`.
3.  **Create Visualization**: Implement `TrainingVisualizer`.
4.  **Create Trainer Wrapper**: Implement `ORPOTrainerWrapper` using `trl`.
5.  **Create Orchestrator**: Implement `ORPOFineTuner`.
6.  **Create CLI**: Implement entry point.

## Dependencies
- `trl`
- `transformers`
- `datasets`
- `peft`
- `matplotlib` (for visualization)
- `pandas`

## Notes
- We will reuse `finetuning/config/base_config.py` for general settings.
- We will reuse `finetuning/core/tokenizer_manager.py` and `finetuning/core/model_loader.py` if possible, or adapt them.
- The `orpo_generator.py` currently in `finetuning/utils/` might be relevant or can be ignored/refactored.
