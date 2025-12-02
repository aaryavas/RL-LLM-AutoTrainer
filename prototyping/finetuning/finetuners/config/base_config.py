"""
Base configuration classes for VB-LoRA fine-tuning.
Following OOP-first design principles with dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Configuration for data processing and splitting."""

    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    text_column: str = "text"
    label_column: str = "label"
    max_length: int = 512
    source_max_len: int = 512
    target_max_len: int = 128

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        if not 0 < self.val_size < 1:
            raise ValueError(f"val_size must be between 0 and 1, got {self.val_size}")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    num_train_epochs: int = 3
    learning_rate: float = 5e-4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 40
    eval_steps: Optional[int] = None
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 2
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "constant"
    group_by_length: bool = True
    seed: int = 42

    def validate(self) -> None:
        """Validate training configuration."""
        if self.num_train_epochs <= 0:
            raise ValueError(f"num_train_epochs must be positive, got {self.num_train_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.per_device_train_batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.per_device_train_batch_size}")


@dataclass
class OutputConfig:
    """Configuration for output and logging."""

    output_dir: str = "./output/vblora_models"
    run_name: Optional[str] = None
    logging_dir: Optional[str] = None
    report_to: str = "none"
    save_label_mapping: bool = True
    save_training_metrics: bool = True
    disable_tqdm: bool = False

    def get_run_name(self) -> str:
        """Generate run name if not provided."""
        if self.run_name:
            return self.run_name
        from datetime import datetime
        return f"vblora_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


@dataclass
class HardwareConfig:
    """Configuration for hardware and device settings."""

    bits: int = 4
    double_quant: bool = True
    quant_type: str = "nf4"
    max_memory_MB: int = 80000
    device_map: str = "auto"
    use_auth_token: bool = False
    trust_remote_code: bool = False

    def validate(self) -> None:
        """Validate hardware configuration."""
        if self.bits not in [4, 8, 16, 32]:
            raise ValueError(f"bits must be one of [4, 8, 16, 32], got {self.bits}")
        if self.quant_type not in ["fp4", "nf4"]:
            raise ValueError(f"quant_type must be 'fp4' or 'nf4', got {self.quant_type}")
