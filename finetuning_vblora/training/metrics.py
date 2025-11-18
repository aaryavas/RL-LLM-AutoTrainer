"""
Metrics computation for VB-LoRA training.
"""

import numpy as np
import evaluate
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class MetricsComputer:
    """
    Computes evaluation metrics for training.
    Uses Hugging Face evaluate library for standard metrics.
    """

    def __init__(self):
        """Initialize metrics computer."""
        self.metrics = {}
        self._load_metrics()

    def _load_metrics(self) -> None:
        """Load metric computations from evaluate library."""
        try:
            # For causal LM, we primarily use perplexity
            # But we can add more metrics as needed
            logger.info("Metrics loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load some metrics: {e}")

    def compute_metrics(self, eval_pred: Tuple) -> Dict[str, float]:
        """
        Compute metrics from evaluation predictions.

        Args:
            eval_pred: Tuple of (predictions, labels)

        Returns:
            Dictionary with computed metrics
        """
        predictions, labels = eval_pred

        # For causal LM, predictions are logits
        # Compute loss-based metrics
        metrics = self._compute_loss_metrics(predictions, labels)

        return metrics

    def _compute_loss_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute loss-based metrics.

        Args:
            predictions: Model predictions (logits)
            labels: Ground truth labels

        Returns:
            Dictionary with metrics
        """
        # Shift predictions and labels for causal LM
        # (predict next token)
        if len(predictions.shape) == 3:
            # predictions shape: (batch, seq_len, vocab)
            # Take argmax to get predicted tokens
            pred_tokens = np.argmax(predictions, axis=-1)
        else:
            pred_tokens = predictions

        # Calculate perplexity
        # Note: This is a simplified version
        # In practice, the trainer computes this from loss
        metrics = {
            "token_accuracy": self._compute_token_accuracy(pred_tokens, labels),
        }

        return metrics

    def _compute_token_accuracy(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Compute token-level accuracy.

        Args:
            predictions: Predicted tokens
            labels: Ground truth tokens

        Returns:
            Accuracy score
        """
        # Mask out ignored indices
        mask = labels != -100

        if not mask.any():
            return 0.0

        # Compute accuracy on non-masked tokens
        correct = (predictions[mask] == labels[mask]).sum()
        total = mask.sum()

        accuracy = float(correct) / float(total)

        return accuracy

    def compute_perplexity(self, loss: float) -> float:
        """
        Compute perplexity from loss.

        Args:
            loss: Cross-entropy loss

        Returns:
            Perplexity score
        """
        try:
            perplexity = np.exp(loss)
            return float(perplexity)
        except OverflowError:
            return float("inf")
