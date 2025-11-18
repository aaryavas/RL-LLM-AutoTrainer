"""
Custom callbacks for VB-LoRA training.
"""

import os
from pathlib import Path
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import logging

logger = logging.getLogger(__name__)


class SavePeftModelCallback(TrainerCallback):
    """
    Callback to save PEFT (VB-LoRA) adapters during training.
    Only saves the adapter weights, not the full model.
    """

    def __init__(self):
        """Initialize callback."""
        super().__init__()

    def on_save(self, args, state, control, **kwargs):
        """
        Called when saving a checkpoint.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            **kwargs: Additional arguments
        """
        self._save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """
        Called at the end of training.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            **kwargs: Additional arguments
        """
        # Create completion marker
        completion_file = Path(args.output_dir) / "completed"
        completion_file.touch()

        # Save final model
        self._save_model(args, state, kwargs)
        return control

    def _save_model(self, args, state, kwargs):
        """
        Save PEFT model checkpoint.

        Args:
            args: Training arguments
            state: Trainer state
            kwargs: Additional arguments containing model
        """
        logger.info("Saving PEFT checkpoint...")

        # Determine checkpoint folder
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint,
                "adapter_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")

        # Save adapter
        model = kwargs.get("model")
        if model is not None:
            model.save_pretrained(peft_model_path)
            logger.info(f"Saved PEFT checkpoint to {peft_model_path}")

            # Remove pytorch_model.bin if it exists (we only want adapter)
            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)


class EpochMetricsCallback(TrainerCallback):
    """
    Callback to display detailed metrics at the end of each epoch.
    Provides visual feedback on training progress.
    """

    def __init__(self, show_metrics: bool = True):
        """
        Initialize callback.

        Args:
            show_metrics: Whether to show detailed metrics
        """
        super().__init__()
        self.show_metrics = show_metrics
        self.epoch_metrics = []

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """
        Called after evaluation at the end of each epoch.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            logs: Training logs
            **kwargs: Additional arguments
        """
        if not self.show_metrics or logs is None:
            return

        # Extract current epoch
        current_epoch = int(state.epoch) if state.epoch is not None else len(self.epoch_metrics) + 1

        # Extract metrics
        metrics = self._extract_metrics(logs)

        # Store metrics
        metrics["epoch"] = current_epoch
        self.epoch_metrics.append(metrics)

        # Display metrics
        self._display_metrics(metrics)

        return control

    def on_train_end(self, args, state, control, logs=None, **kwargs):
        """
        Called at the end of training to show summary.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            logs: Training logs
            **kwargs: Additional arguments
        """
        if not self.show_metrics or not self.epoch_metrics:
            return

        self._display_summary()

        return control

    def _extract_metrics(self, logs: dict) -> dict:
        """
        Extract relevant metrics from logs.

        Args:
            logs: Training logs dictionary

        Returns:
            Dictionary with extracted metrics
        """
        return {
            "eval_loss": logs.get("eval_loss", 0.0),
            "train_loss": logs.get("train_loss", 0.0),
            "learning_rate": logs.get("learning_rate", 0.0),
        }

    def _display_metrics(self, metrics: dict) -> None:
        """
        Display metrics for an epoch.

        Args:
            metrics: Dictionary with metrics
        """
        epoch = metrics["epoch"]
        eval_loss = metrics["eval_loss"]
        train_loss = metrics["train_loss"]
        lr = metrics["learning_rate"]

        print(f"\n{'='*50}")
        print(f"EPOCH {epoch} RESULTS")
        print(f"{'='*50}")
        print(f"  Training Loss:   {train_loss:.4f}")
        print(f"  Validation Loss: {eval_loss:.4f}")
        print(f"  Learning Rate:   {lr:.2e}")
        print(f"{'='*50}\n")

    def _display_summary(self) -> None:
        """Display training summary."""
        print(f"\n{'='*50}")
        print("TRAINING SUMMARY")
        print(f"{'='*50}")
        print(f"  Total Epochs: {len(self.epoch_metrics)}")

        # Find best epoch
        best_idx = min(
            range(len(self.epoch_metrics)),
            key=lambda i: self.epoch_metrics[i]["eval_loss"]
        )
        best_metrics = self.epoch_metrics[best_idx]

        print(f"  Best Epoch: {best_metrics['epoch']}")
        print(f"  Best Validation Loss: {best_metrics['eval_loss']:.4f}")

        # Show improvement
        if len(self.epoch_metrics) > 1:
            first_loss = self.epoch_metrics[0]["eval_loss"]
            final_loss = self.epoch_metrics[-1]["eval_loss"]
            improvement = first_loss - final_loss

            print(f"  Loss Improvement: {improvement:+.4f}")

        print(f"{'='*50}\n")
