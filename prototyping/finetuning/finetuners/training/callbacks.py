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
            "bleu": logs.get("eval_bleu", 0.0),
            "rouge1": logs.get("eval_rouge1", 0.0),
            "rouge2": logs.get("eval_rouge2", 0.0),
            "rougeL": logs.get("eval_rougeL", 0.0),
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
        bleu = metrics["bleu"]
        rouge1 = metrics["rouge1"]
        rouge2 = metrics["rouge2"]
        rougeL = metrics["rougeL"]

        print(f"\n{'='*60}")
        print(f"EPOCH {epoch} RESULTS")
        print(f"{'='*60}")
        print(f"  Training Loss:      {train_loss:.4f}")
        print(f"  Validation Loss:    {eval_loss:.4f}")
        print(f"  Learning Rate:      {lr:.2e}")
        print(f"  {'─'*56}")
        print(f"  Text Generation Metrics:")
        print(f"    BLEU Score:       {bleu:.4f}")
        print(f"    ROUGE-1:          {rouge1:.4f}")
        print(f"    ROUGE-2:          {rouge2:.4f}")
        print(f"    ROUGE-L:          {rougeL:.4f}")
        print(f"{'='*60}\n")

    def _display_summary(self) -> None:
        """Display training summary."""
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"  Total Epochs: {len(self.epoch_metrics)}")

        # Find best epoch by validation loss
        best_loss_idx = min(
            range(len(self.epoch_metrics)),
            key=lambda i: self.epoch_metrics[i]["eval_loss"]
        )
        best_loss_metrics = self.epoch_metrics[best_loss_idx]

        print(f"\n  Best Model by Validation Loss:")
        print(f"    Epoch:            {best_loss_metrics['epoch']}")
        print(f"    Validation Loss:  {best_loss_metrics['eval_loss']:.4f}")
        print(f"    BLEU Score:       {best_loss_metrics['bleu']:.4f}")
        print(f"    ROUGE-L:          {best_loss_metrics['rougeL']:.4f}")

        # Find best epoch by BLEU score
        if any(m['bleu'] > 0 for m in self.epoch_metrics):
            best_bleu_idx = max(
                range(len(self.epoch_metrics)),
                key=lambda i: self.epoch_metrics[i]["bleu"]
            )
            best_bleu_metrics = self.epoch_metrics[best_bleu_idx]

            print(f"\n  Best Model by BLEU Score:")
            print(f"    Epoch:            {best_bleu_metrics['epoch']}")
            print(f"    BLEU Score:       {best_bleu_metrics['bleu']:.4f}")
            print(f"    ROUGE-1:          {best_bleu_metrics['rouge1']:.4f}")
            print(f"    ROUGE-2:          {best_bleu_metrics['rouge2']:.4f}")
            print(f"    ROUGE-L:          {best_bleu_metrics['rougeL']:.4f}")

        # Show improvement
        if len(self.epoch_metrics) > 1:
            first_loss = self.epoch_metrics[0]["eval_loss"]
            final_loss = self.epoch_metrics[-1]["eval_loss"]
            improvement = first_loss - final_loss

            print(f"\n  Overall Improvement:")
            print(f"    Loss Reduction:   {improvement:+.4f}")

            if self.epoch_metrics[-1]['bleu'] > 0:
                first_bleu = self.epoch_metrics[0]["bleu"]
                final_bleu = self.epoch_metrics[-1]["bleu"]
                bleu_improvement = final_bleu - first_bleu
                print(f"    BLEU Gain:        {bleu_improvement:+.4f}")

        print(f"{'='*60}\n")
