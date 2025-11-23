"""
VB-LoRA specific configuration.
VB-LoRA extends LoRA with vector banks for more efficient adaptation.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VBLoRAConfig:
    """Configuration for VB-LoRA specific parameters."""

    # Core VB-LoRA parameters
    lora_r: int = 4  # LoRA rank
    lora_alpha: float = 16  # LoRA scaling factor
    lora_dropout: float = 0.05
    num_vectors: int = 90  # Number of vectors in the vector bank
    vector_length: int = 64  # Changed from 160 -> 64 (must be multiple of 64, divides all SmolLM2 dims)
    save_only_topk_weights: bool = True

    # Learning rates for different parameter groups
    learning_rate_vector_bank: float = 1e-3
    learning_rate_logits: float = 1e-2
    learning_rate_base: float = 2e-4

    # Target modules for LoRA adaptation
    target_modules: Optional[List[str]] = None

    def __post_init__(self):
        """Set default target modules if not provided."""
        if self.target_modules is None:
            # Default to all linear layers
            self.target_modules = ["all"]

    def validate(self) -> None:
        """Validate VB-LoRA configuration."""
        if self.lora_r <= 0:
            raise ValueError(f"lora_r must be positive, got {self.lora_r}")
        if self.num_vectors <= 0:
            raise ValueError(f"num_vectors must be positive, got {self.num_vectors}")
        if self.vector_length <= 0:
            raise ValueError(f"vector_length must be positive, got {self.vector_length}")

    def to_peft_config_dict(self) -> dict:
        """Convert to dictionary for PEFT VBLoRAConfig."""
        return {
            "r": self.lora_r,
            "vblora_dropout": self.lora_dropout,
            "num_vectors": self.num_vectors,
            "vector_length": self.vector_length,
            "save_only_topk_weights": self.save_only_topk_weights,
            "task_type": "CAUSAL_LM",
        }
@dataclass
class VBLoRADefaults:
    """Default configurations for different use cases."""

    @staticmethod
    def quick_test() -> VBLoRAConfig:
        return VBLoRAConfig(
            lora_r=4,
            num_vectors=32,
            vector_length=64,  # Changed from 128
        )

    @staticmethod
    def standard() -> VBLoRAConfig:
        return VBLoRAConfig(
            lora_r=4,
            num_vectors=90,
            vector_length=64,  # Changed from 160
        )

    @staticmethod
    def high_capacity() -> VBLoRAConfig:
        return VBLoRAConfig(
            lora_r=8,
            num_vectors=2048,
            vector_length=64,  # Changed from 160
        )

    @staticmethod
    def memory_efficient() -> VBLoRAConfig:
        return VBLoRAConfig(
            lora_r=4,
            num_vectors=64,
            vector_length=64,  # Changed from 128
        )