"""
Configuration classes for PEFT fine-tuning framework.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from transformers import TrainingArguments


@dataclass
class ModelConfig:
    """Configuration for the base model."""
    model_name: str = "HuggingFaceTB/SmolLM-1.7B-Instruct"
    device_map: str = "auto"
    torch_dtype: Optional[str] = None
    trust_remote_code: bool = True
    use_cache: bool = False


@dataclass
class PEFTConfig:
    """Configuration for PEFT adaptation."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    modules_to_save: List[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """Configuration for data preprocessing."""
    dataset_name: str = "timdettmers/openassistant-guanaco"
    text_column: str = "text"
    max_length: int = 2048
    train_split: str = "train"
    eval_split: str = "test"
    test_size: float = 0.1
    preprocessing_num_workers: int = 4


@dataclass
class TrainingConfig:
    """Configuration for training setup."""
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = True
    bf16: bool = False
    dataloader_pin_memory: bool = False


@dataclass
class IterativeTrainingConfig:
    """Configuration for iterative training process."""
    max_iterations: int = 3
    iteration_save_prefix: str = "iteration_"
    enable_checkpoint_resume: bool = True
    log_iteration_metrics: bool = True
    hyperparameter_search: bool = False
    hyperparameter_configs: List[Dict[str, Any]] = field(default_factory=list)

    # Auto-tuning options
    auto_tune_lr: bool = False
    lr_range: List[float] = field(default_factory=lambda: [1e-5, 5e-4])
    auto_tune_r: bool = False
    r_range: List[int] = field(default_factory=lambda: [8, 32])


@dataclass
class FrameworkConfig:
    """Main configuration class combining all settings."""
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    iterative: IterativeTrainingConfig = field(default_factory=IterativeTrainingConfig)

    def to_training_arguments(self) -> TrainingArguments:
        """Convert training config to HuggingFace TrainingArguments."""
        args_config = {
            k: v for k, v in self.training.__dict__.items()
            if not k.startswith('_')
        }
        return TrainingArguments(**args_config)

    def update_for_iteration(self, iteration: int) -> None:
        """Update configuration settings for a specific iteration."""
        if self.iterative.hyperparameter_search and iteration < len(self.iterative.hyperparameter_configs):
            # Update based on predefined hyperparameter configs
            hp_config = self.iterative.hyperparameter_configs[iteration - 1]
            for key, value in hp_config.items():
                if hasattr(self.training, key):
                    setattr(self.training, key, value)
                elif hasattr(self.peft, key):
                    setattr(self.peft, key, value)

        # Update output directory to include iteration
        self.training.output_dir = f"{self.iterative.iteration_save_prefix}{iteration:02d}"

    def get_experiment_name(self) -> str:
        """Generate a unique experiment name based on config."""
        return "02d"
