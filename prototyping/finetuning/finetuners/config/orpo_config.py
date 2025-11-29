"""
ORPO-specific configuration classes.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ORPOSpecificConfig:
    """Configuration specific to ORPO training."""
    
    beta: float = 0.5
    max_prompt_length: int = 512
    max_completion_length: int = 1024
    disable_dropout: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.max_prompt_length <= 0:
            raise ValueError(f"max_prompt_length must be positive, got {self.max_prompt_length}")
        if self.max_completion_length <= 0:
            raise ValueError(f"max_completion_length must be positive, got {self.max_completion_length}")
