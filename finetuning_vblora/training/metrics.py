"""
Metrics computation for VB-LoRA training.
Enhanced with precision, recall, and F1 score for classification tasks.
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsComputer:
    """
    Computes evaluation metrics for training.
    Includes classification metrics: precision, recall, F1 score.
    """

    def __init__(self, tokenizer=None, label_list=None):
        """
        Initialize metrics computer.
        
        Args:
            tokenizer: Tokenizer for decoding predictions (optional)
            label_list: List of unique labels for classification (optional)
        """
        self.tokenizer = tokenizer
        self.label_list = label_list
        logger.info("Metrics computer initialized with classification metrics")

    def compute_metrics(self, eval_pred: Tuple) -> Dict[str, float]:
        """
        Compute metrics from evaluation predictions.
        Includes token accuracy, precision, recall, and F1 score.

        Args:
            eval_pred: Tuple of (predictions, labels)

        Returns:
            Dictionary with computed metrics
        """
        predictions, labels = eval_pred

        # Initialize metrics dictionary
        metrics = {}

        # Compute token-level metrics
        token_metrics = self._compute_token_metrics(predictions, labels)
        metrics.update(token_metrics)

        # If tokenizer is available, compute classification metrics
        if self.tokenizer is not None:
            try:
                classification_metrics = self._compute_classification_metrics(
                    predictions, labels
                )
                metrics.update(classification_metrics)
            except Exception as e:
                logger.warning(f"Could not compute classification metrics: {e}")

        return metrics

    def _compute_token_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute token-level metrics.

        Args:
            predictions: Model predictions (logits)
            labels: Ground truth labels

        Returns:
            Dictionary with token metrics
        """
        # Get predicted tokens
        if len(predictions.shape) == 3:
            # predictions shape: (batch, seq_len, vocab)
            pred_tokens = np.argmax(predictions, axis=-1)
        else:
            pred_tokens = predictions

        # Calculate token accuracy
        token_accuracy = self._compute_token_accuracy(pred_tokens, labels)

        return {
            "token_accuracy": token_accuracy,
        }

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

    def _compute_classification_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute classification metrics: precision, recall, F1.
        
        This extracts the predicted and true labels from the sequences
        and computes standard classification metrics.

        Args:
            predictions: Model predictions (logits)
            labels: Ground truth labels

        Returns:
            Dictionary with classification metrics
        """
        # Get predicted tokens
        if len(predictions.shape) == 3:
            pred_tokens = np.argmax(predictions, axis=-1)
        else:
            pred_tokens = predictions

        # Extract label sequences (where labels != -100)
        pred_labels = []
        true_labels = []

        for i in range(len(labels)):
            # Find non-ignored positions (these are the output/label tokens)
            mask = labels[i] != -100
            
            if mask.any():
                # Get the predicted and true label tokens
                pred_seq = pred_tokens[i][mask]
                true_seq = labels[i][mask]
                
                # Decode to get text
                pred_text = self.tokenizer.decode(pred_seq, skip_special_tokens=True).strip()
                true_text = self.tokenizer.decode(true_seq, skip_special_tokens=True).strip()
                
                pred_labels.append(pred_text)
                true_labels.append(true_text)

        if not pred_labels:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }

        # Compute classification metrics
        try:
            # Use macro averaging for multi-class
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels,
                pred_labels,
                average='macro',
                zero_division=0
            )
            
            # Also compute accuracy
            accuracy = accuracy_score(true_labels, pred_labels)

            return {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(accuracy),
            }
        except Exception as e:
            logger.warning(f"Error computing classification metrics: {e}")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }

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