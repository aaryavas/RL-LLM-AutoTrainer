"""
Model variant configurations for different SmolLM2 models.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelVariantConfig:
    """Configuration for a specific model variant."""

    model_name: str
    batch_size: int
    learning_rate: float
    recommended_num_vectors: int
    description: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_vectors": self.recommended_num_vectors,
        }


# SmolLM2 variant configurations
SMOLLM2_VARIANTS: Dict[str, ModelVariantConfig] = {
    "SmolLM2-135M": ModelVariantConfig(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        batch_size=16,
        learning_rate=5e-4,
        recommended_num_vectors=64,
        description="Smallest SmolLM2 variant (135M parameters)",
    ),
    "SmolLM2-360M": ModelVariantConfig(
        model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        batch_size=12,
        learning_rate=3e-4,
        recommended_num_vectors=90,
        description="Medium SmolLM2 variant (360M parameters)",
    ),
    "SmolLM2-1.7B": ModelVariantConfig(
        model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        batch_size=4,
        learning_rate=2e-4,
        recommended_num_vectors=128,
        description="Largest SmolLM2 variant (1.7B parameters)",
    ),
}


# Preset configurations
class PresetConfigs:
    """Preset training configurations."""

    @staticmethod
    def quick_test() -> dict:
        """Quick test configuration."""
        return {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "save_strategy": "no",
            "eval_strategy": "no",
        }

    @staticmethod
    def standard() -> dict:
        """Standard training configuration."""
        return {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "learning_rate": 2e-4,
            "gradient_accumulation_steps": 4,
        }

    @staticmethod
    def thorough() -> dict:
        """Thorough training configuration."""
        return {
            "num_train_epochs": 5,
            "per_device_train_batch_size": 4,
            "learning_rate": 1e-4,
            "gradient_accumulation_steps": 4,
            "early_stopping_patience": 3,
        }

    @staticmethod
    def memory_efficient() -> dict:
        """Memory efficient configuration."""
        return {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "gradient_checkpointing": True,
            "bf16": True,
        }


PRESET_CONFIGS: Dict[str, dict] = {
    "quick_test": PresetConfigs.quick_test(),
    "standard": PresetConfigs.standard(),
    "thorough": PresetConfigs.thorough(),
    "memory_efficient": PresetConfigs.memory_efficient(),
}


def get_variant_config(variant_name: str) -> ModelVariantConfig:
    """
    Get configuration for a specific model variant.

    Args:
        variant_name: Name of the variant (e.g., 'SmolLM2-1.7B')

    Returns:
        ModelVariantConfig for the specified variant

    Raises:
        ValueError: If variant name is not recognized
    """
    if variant_name not in SMOLLM2_VARIANTS:
        available = list(SMOLLM2_VARIANTS.keys())
        raise ValueError(f"Unknown variant: {variant_name}. Available: {available}")
    return SMOLLM2_VARIANTS[variant_name]


def get_preset_config(preset_name: str) -> dict:
    """
    Get a preset training configuration.

    Args:
        preset_name: Name of the preset (e.g., 'standard')

    Returns:
        Dictionary with preset configuration

    Raises:
        ValueError: If preset name is not recognized
    """
    if preset_name not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    return PRESET_CONFIGS[preset_name].copy()
