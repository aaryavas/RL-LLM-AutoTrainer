"""
SmolLM2 Fine-tuning with LoRA PEFT

This package provides tools for fine-tuning SmolLM2 models using LoRA (Low-Rank Adaptation) 
PEFT on synthetic text classification data.

Main components:
- SmolLM2FineTuner: Main class for fine-tuning
- split_synthetic_data: Function for data splitting
- Configuration presets and model variants
- CLI interface for easy usage

Example usage:
    >>> from finetuning import SmolLM2FineTuner
    >>> finetuner = SmolLM2FineTuner()
    >>> model_path, results = finetuner.finetune_pipeline("data.csv")
"""

from .finetuning import SmolLM2FineTuner, split_synthetic_data
from .config import (
    MODEL_VARIANTS, 
    PRESETS, 
    get_config_for_variant, 
    get_preset_config,
    merge_configs
)

__version__ = "1.0.0"
__author__ = "RL-LLM-AutoTrainer"
__description__ = "Fine-tuning SmolLM2 models with LoRA PEFT"

__all__ = [
    'SmolLM2FineTuner',
    'split_synthetic_data', 
    'MODEL_VARIANTS',
    'PRESETS',
    'get_config_for_variant',
    'get_preset_config',
    'merge_configs'
]