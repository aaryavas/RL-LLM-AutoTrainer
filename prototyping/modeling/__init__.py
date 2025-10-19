"""
PEFT Fine-tuning Framework for SmolLM
"""

from .config import FrameworkConfig, ModelConfig, PEFTConfig, DataConfig, TrainingConfig, IterativeTrainingConfig
from .peft import PEFTFineTuner, create_default_config, quick_train
from .rl import (
    RLTrainer,
    RLPEFTFineTuner,
    IterationRewardFunction,
    TrainingLossReward,
    EvaluationReward,
    CombinedReward,
    HyperparameterAction,
    TrainingActionSpace,
    create_rl_trainer,
    rl_train
)

__all__ = [
    # Config classes
    "FrameworkConfig",
    "ModelConfig",
    "PEFTConfig",
    "DataConfig",
    "TrainingConfig",
    "IterativeTrainingConfig",

    # PEFT training
    "PEFTFineTuner",
    "create_default_config",
    "quick_train",

    # RL components
    "RLTrainer",
    "RLPEFTFineTuner",
    "IterationRewardFunction",
    "TrainingLossReward",
    "EvaluationReward",
    "CombinedReward",
    "HyperparameterAction",
    "TrainingActionSpace",
    "create_rl_trainer",
    "rl_train",
]
