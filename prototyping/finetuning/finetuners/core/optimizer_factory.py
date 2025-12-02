"""
Custom optimizer factory for VB-LoRA with separate learning rates.
VB-LoRA requires different learning rates for vector bank, logits, and base parameters.
"""

import torch
from torch.optim import Optimizer
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class OptimizerFactory:
    """
    Factory for creating optimizers with parameter-specific learning rates.
    Essential for VB-LoRA which requires separate learning rates for different components.
    """

    def __init__(
        self,
        learning_rate: float = 2e-4,
        learning_rate_vector_bank: float = 1e-3,
        learning_rate_logits: float = 1e-2,
        weight_decay: float = 0.0,
    ):
        """
        Initialize optimizer factory.

        Args:
            learning_rate: Base learning rate for regular parameters
            learning_rate_vector_bank: Learning rate for vector bank parameters
            learning_rate_logits: Learning rate for logits parameters
            weight_decay: Weight decay coefficient
        """
        self.learning_rate = learning_rate
        self.learning_rate_vector_bank = learning_rate_vector_bank
        self.learning_rate_logits = learning_rate_logits
        self.weight_decay = weight_decay

    def create_optimizer(
        self,
        model: torch.nn.Module,
        training_args,
    ) -> Optimizer:
        """
        Create optimizer with parameter groups for VB-LoRA.

        Args:
            model: Model to optimize
            training_args: Training arguments from transformers

        Returns:
            Configured optimizer
        """
        logger.info("Creating optimizer with separate learning rates for VB-LoRA")

        # Get parameter groups
        optimizer_grouped_parameters = self._create_parameter_groups(model)

        # Log parameter group information
        self._log_parameter_groups(optimizer_grouped_parameters)

        # Get optimizer class and kwargs from training args
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            training_args
        )

        # Override learning rate (will be set per group)
        optimizer_kwargs.pop("lr", None)

        # Create optimizer
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        logger.info(f"Created optimizer: {optimizer.__class__.__name__}")

        return optimizer

    def _create_parameter_groups(self, model: torch.nn.Module) -> List[Dict[str, Any]]:
        """
        Create parameter groups with different learning rates.

        Args:
            model: Model to create parameter groups for

        Returns:
            List of parameter group dictionaries
        """
        # Get parameters that should have weight decay
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        # Identify vector bank and logits parameters
        vector_bank_params = self._identify_vector_bank_parameters(model)
        logits_params = self._identify_logits_parameters(model)

        # Create parameter groups
        parameter_groups = [
            # Regular parameters with weight decay
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if n in decay_parameters
                    and n not in logits_params
                    and n not in vector_bank_params
                    and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
                "lr": self.learning_rate,
            },
            # Regular parameters without weight decay
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if n not in decay_parameters
                    and n not in logits_params
                    and n not in vector_bank_params
                    and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": self.learning_rate,
            },
            # Vector bank parameters
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if n in vector_bank_params and p.requires_grad
                ],
                "lr": self.learning_rate_vector_bank,
                "weight_decay": 0.0,
            },
            # Logits parameters
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if n in logits_params and p.requires_grad
                ],
                "lr": self.learning_rate_logits,
                "weight_decay": 0.0,
            },
        ]

        return parameter_groups

    def _identify_vector_bank_parameters(self, model: torch.nn.Module) -> List[str]:
        """
        Identify parameters belonging to the vector bank.

        Args:
            model: Model to search

        Returns:
            List of parameter names
        """
        vector_bank_params = [
            name for name, _ in model.named_parameters()
            if "vector_bank" in name
        ]
        return vector_bank_params

    def _identify_logits_parameters(self, model: torch.nn.Module) -> List[str]:
        """
        Identify parameters belonging to logits layers.

        Args:
            model: Model to search

        Returns:
            List of parameter names
        """
        logits_params = [
            name for name, _ in model.named_parameters()
            if "logits" in name
        ]
        return logits_params

    def _log_parameter_groups(self, parameter_groups: List[Dict[str, Any]]) -> None:
        """
        Log information about parameter groups.

        Args:
            parameter_groups: List of parameter group dictionaries
        """
        group_names = ["Base (with decay)", "Base (no decay)", "Vector Bank", "Logits"]

        for i, (group, name) in enumerate(zip(parameter_groups, group_names)):
            num_params = sum(p.numel() for p in group["params"])
            lr = group.get("lr", "default")
            wd = group.get("weight_decay", 0.0)

            logger.info(
                f"Parameter Group {i} ({name}): "
                f"{num_params:,} params, lr={lr}, weight_decay={wd}"
            )
