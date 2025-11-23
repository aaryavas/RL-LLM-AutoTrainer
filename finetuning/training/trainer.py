"""
VB-LoRA trainer orchestration.
"""

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from typing import Optional, Dict, Any
from pathlib import Path
import logging

from training.callbacks import SavePeftModelCallback, EpochMetricsCallback
from training.metrics import MetricsComputer

logger = logging.getLogger(__name__)


class VBLoRATrainer:
    """
    Orchestrates VB-LoRA fine-tuning training process.
    Wraps Hugging Face Trainer with VB-LoRA specific configuration.
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        data_collator,
        optimizer,
        training_config: Dict[str, Any],
        output_config: Dict[str, Any],
    ):
        """
        Initialize VB-LoRA trainer.

        Args:
            model: Model to train
            tokenizer: Tokenizer instance
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator
            optimizer: Optimizer instance
            training_config: Training configuration dictionary
            output_config: Output configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.optimizer = optimizer
        self.training_config = training_config
        self.output_config = output_config

        self.trainer = None
        # Pass tokenizer to metrics computer for BLEU/ROUGE computation
        self.metrics_computer = MetricsComputer(tokenizer=tokenizer)

    def setup_trainer(self, show_epoch_metrics: bool = True) -> None:
        """
        Setup the Hugging Face Trainer with all configurations.

        Args:
            show_epoch_metrics: Whether to show detailed metrics per epoch
        """
        logger.info("Setting up trainer")

        # Create output directory
        output_dir = Path(self.output_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create training arguments
        training_args = self._create_training_arguments()

        # Setup callbacks
        callbacks = self._create_callbacks(show_epoch_metrics)

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            optimizers=(self.optimizer, None),
            callbacks=callbacks,
            compute_metrics=self.metrics_computer.compute_metrics,
        )

        logger.info("Trainer setup complete")

    def train(self) -> Dict[str, Any]:
        """
        Run training.

        Returns:
            Dictionary with training metrics
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not setup. Call setup_trainer() first.")

        logger.info("Starting training")

        # Disable model cache during training
        self.model.config.use_cache = False

        # Train
        train_result = self.trainer.train()

        # Save final model
        self.trainer.save_model()
        self.trainer.save_state()

        logger.info("Training completed")

        return train_result.metrics

    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation on eval dataset.

        Returns:
            Dictionary with evaluation metrics
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not setup. Call setup_trainer() first.")

        logger.info("Running evaluation")

        eval_metrics = self.trainer.evaluate()

        logger.info(f"Evaluation complete: {eval_metrics}")

        return eval_metrics

    def _create_training_arguments(self) -> TrainingArguments:
        """
        Create TrainingArguments from configuration.

        Returns:
            TrainingArguments instance
        """
        run_name = self.output_config.get("run_name")
        if not run_name:
            # Generate run name if not provided
            from datetime import datetime
            run_name = f"vblora_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return TrainingArguments(
            output_dir=self.output_config["output_dir"],
            run_name=run_name,
            num_train_epochs=self.training_config["num_train_epochs"],
            per_device_train_batch_size=self.training_config["per_device_train_batch_size"],
            per_device_eval_batch_size=self.training_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.training_config["gradient_accumulation_steps"],
            warmup_steps=self.training_config["warmup_steps"],
            weight_decay=self.training_config["weight_decay"],
            logging_steps=self.training_config["logging_steps"],
            save_steps=self.training_config["save_steps"],
            save_total_limit=self.training_config["save_total_limit"],
            eval_strategy=self.training_config["eval_strategy"],
            save_strategy=self.training_config["save_strategy"],
            load_best_model_at_end=self.training_config["load_best_model_at_end"],
            metric_for_best_model=self.training_config["metric_for_best_model"],
            greater_is_better=self.training_config["greater_is_better"],
            fp16=self.training_config["fp16"],
            bf16=self.training_config["bf16"],
            gradient_checkpointing=self.training_config["gradient_checkpointing"],
            optim=self.training_config["optim"],
            lr_scheduler_type=self.training_config["lr_scheduler_type"],
            warmup_ratio=self.training_config["warmup_ratio"],
            max_grad_norm=self.training_config["max_grad_norm"],
            group_by_length=self.training_config["group_by_length"],
            seed=self.training_config["seed"],
            report_to=self.output_config.get("report_to", "none"),
            logging_dir=self.output_config.get("logging_dir"),
            disable_tqdm=self.output_config.get("disable_tqdm", False),
            remove_unused_columns=False,
        )

    def _create_callbacks(self, show_epoch_metrics: bool) -> list:
        """
        Create trainer callbacks.

        Args:
            show_epoch_metrics: Whether to show detailed metrics

        Returns:
            List of callback instances
        """
        callbacks = []

        # Add PEFT saving callback
        callbacks.append(SavePeftModelCallback())

        # Add early stopping if configured
        if self.training_config.get("early_stopping_patience"):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config["early_stopping_patience"]
                )
            )

        # Add epoch metrics callback
        if show_epoch_metrics:
            callbacks.append(EpochMetricsCallback(show_metrics=True))

        return callbacks

    def get_trainer(self) -> Trainer:
        """
        Get the underlying Hugging Face Trainer.

        Returns:
            Trainer instance
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not setup. Call setup_trainer() first.")

        return self.trainer
