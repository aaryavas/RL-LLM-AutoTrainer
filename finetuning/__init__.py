"""
VB-LoRA Fine-tuning Module for SmolLM2 and other language models.

This module provides a complete pipeline for fine-tuning language models
using VB-LoRA (Vector Bank LoRA) with efficient memory usage through quantization.
"""

from .finetuning import SmolLM2VBLoRAFineTuner
from .config import (
    DataConfig,
    TrainingConfig,
    VBLoRAConfig,
    OutputConfig,
    HardwareConfig,
    SMOLLM2_VARIANTS,
    PRESET_CONFIGS,
)
from .utils import DataSplitter, ensure_dir, save_json, load_json

__version__ = "1.0.0"

__all__ = [
    "SmolLM2VBLoRAFineTuner",
    "DataConfig",
    "TrainingConfig",
    "VBLoRAConfig",
    "OutputConfig",
    "HardwareConfig",
    "SMOLLM2_VARIANTS",
    "PRESET_CONFIGS",
    "DataSplitter",
    "ensure_dir",
    "save_json",
    "load_json",
]
