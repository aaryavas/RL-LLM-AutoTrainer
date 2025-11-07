"""
Configuration file for SmolLM2 fine-tuning.
This file contains default settings that can be customized for different fine-tuning runs.
"""

# Model configuration
MODEL_CONFIG = {
    "base_model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "max_length": 512,
    "model_cache_dir": None,  # Use default cache
}

# LoRA configuration  
LORA_CONFIG = {
    "r": 16,  # Rank - higher values = more parameters but better adaptation
    "lora_alpha": 32,  # LoRA scaling parameter
    "lora_dropout": 0.1,  # Dropout for LoRA layers
    "target_modules": [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj", 
        "down_proj",
    ],
    "bias": "none",  # Whether to train bias parameters
    "task_type": "SEQ_CLS",  # Task type for PEFT
}

# Data configuration
DATA_CONFIG = {
    "test_size": 0.2,  # Proportion for test set
    "val_size": 0.1,   # Proportion of training data for validation
    "random_state": 42,  # Random seed for reproducibility
    "text_column": "text",  # Name of text column in CSV
    "label_column": "label",  # Name of label column in CSV
}

# Training configuration
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch", 
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_f1",
    "greater_is_better": True,
    "early_stopping_patience": 2,
    "logging_steps": 10,
    "seed": 42,
    "fp16": True,  # Use mixed precision training
    "dataloader_pin_memory": False,
    "gradient_accumulation_steps": 1,
    "dataloader_num_workers": 0,  # Set to 0 to avoid multiprocessing issues
}

# Output configuration
OUTPUT_CONFIG = {
    "output_dir": "./finetuned_models",
    "run_name_prefix": "smollm2_finetune",
    "save_label_mapping": True,
    "save_training_metrics": True,
}

# Hardware configuration
HARDWARE_CONFIG = {
    "use_cuda": True,  # Whether to use CUDA if available
    "device_map": "auto",  # Device mapping strategy
    "torch_dtype": "auto",  # Data type for model weights
}

# Advanced training options
ADVANCED_CONFIG = {
    "gradient_checkpointing": False,  # Trade memory for compute
    "optim": "adamw_hf",  # Optimizer
    "lr_scheduler_type": "linear",  # Learning rate scheduler
    "warmup_ratio": 0.1,  # Warmup ratio instead of steps
    "max_grad_norm": 1.0,  # Gradient clipping
    "label_smoothing_factor": 0.0,  # Label smoothing
}

# Evaluation configuration
EVAL_CONFIG = {
    "eval_accumulation_steps": None,  # Steps to accumulate before moving to CPU
    "eval_delay": 0,  # Delay evaluation until this many steps
    "include_inputs_for_metrics": False,  # Include inputs in metric computation
}

# Logging and monitoring
LOGGING_CONFIG = {
    "report_to": None,  # Don't use wandb/tensorboard by default
    "log_level": "info",
    "disable_tqdm": False,
    "dataloader_drop_last": False,
}

# Memory optimization
MEMORY_CONFIG = {
    "max_memory_MB": None,  # Maximum memory per GPU
    "low_cpu_mem_usage": True,  # Use low CPU memory when loading model
    "use_mps_device": False,  # Use MPS on Apple Silicon
}

# Model-specific configurations for different SmolLM2 variants
MODEL_VARIANTS = {
    "SmolLM2-135M": {
        "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct", 
        "batch_size": 16,
        "learning_rate": 5e-4,
    },
    "SmolLM2-360M": {
        "model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "batch_size": 12,
        "learning_rate": 3e-4,
    },
    "SmolLM2-1.7B": {
        "model_name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "batch_size": 8,
        "learning_rate": 2e-4,
    },
}

# Quick preset configurations
PRESETS = {
    "quick_test": {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 4,
        "save_strategy": "no",
        "evaluation_strategy": "no",
    },
    "standard": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "learning_rate": 2e-4,
    },
    "thorough": {
        "num_train_epochs": 5,
        "per_device_train_batch_size": 4,
        "learning_rate": 1e-4,
        "early_stopping_patience": 3,
    },
    "memory_efficient": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "fp16": True,
    }
}

def get_config_for_variant(variant_name: str) -> dict:
    """Get configuration for a specific SmolLM2 variant."""
    if variant_name not in MODEL_VARIANTS:
        raise ValueError(f"Unknown variant: {variant_name}. Available: {list(MODEL_VARIANTS.keys())}")
    
    config = MODEL_VARIANTS[variant_name].copy()
    return config

def get_preset_config(preset_name: str) -> dict:
    """Get a preset configuration."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    
    return PRESETS[preset_name].copy()

def merge_configs(*configs) -> dict:
    """Merge multiple configuration dictionaries."""
    merged = {}
    for config in configs:
        merged.update(config)
    return merged