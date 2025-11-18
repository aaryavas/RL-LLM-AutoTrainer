"""
Metrics computation for VB-LoRA training.
Enhanced with BLEU and ROUGE metrics for text generation quality.
"""

import numpy as np
import evaluate
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsComputer:
    """
    Computes evaluation metrics for training.
    Uses Hugging Face evaluate library for standard metrics.
    Includes BLEU and ROUGE metrics for text generation quality assessment.
    """

    def __init__(self, tokenizer=None):
        """
        Initialize metrics computer.

        Args:
            tokenizer: Tokenizer for decoding predictions (required for BLEU/ROUGE)
        """
        self.tokenizer = tokenizer
        self.metrics = {}
        self._load_metrics()

    def _load_metrics(self) -> None:
        """Load metric computations from evaluate library."""
        try:
            # Load BLEU metric for machine translation quality
            self.metrics['bleu'] = evaluate.load('bleu')
            logger.info("BLEU metric loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load BLEU metric: {e}")
            self.metrics['bleu'] = None

        try:
            # Load ROUGE metric for text generation/summarization quality
            self.metrics['rouge'] = evaluate.load('rouge')
            logger.info("ROUGE metric loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load ROUGE metric: {e}")
            self.metrics['rouge'] = None

    def compute_metrics(self, eval_pred: Tuple) -> Dict[str, float]:
        """
        Compute metrics from evaluation predictions.
        Includes token accuracy, BLEU, and ROUGE scores.

        Args:
            eval_pred: Tuple of (predictions, labels)

        Returns:
            Dictionary with computed metrics
        """
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        # For causal LM, predictions are logits
        # Compute loss-based metrics
        metrics = self._compute_loss_metrics(predictions, labels)

        # Compute BLEU and ROUGE if tokenizer is available
        if self.tokenizer is not None:
            try:
                text_metrics = self._compute_text_generation_metrics(predictions, labels)
                metrics.update(text_metrics)
            except Exception as e:
                logger.warning(f"Could not compute text generation metrics: {e}")

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

    def _compute_text_generation_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute BLEU and ROUGE metrics for text generation quality.

        Args:
            predictions: Model predictions (logits)
            labels: Ground truth labels

        Returns:
            Dictionary with BLEU and ROUGE metrics
        """
        # Get predicted tokens
        if len(predictions.shape) == 3:
            pred_tokens = np.argmax(predictions, axis=-1)
        else:
            pred_tokens = predictions

        # Decode predictions and labels to text
        pred_texts = []
        label_texts = []

        for i in range(len(labels)):
            # Find non-ignored positions
            mask = labels[i] != -100

            if mask.any():
                # Get the predicted and true sequences
                pred_seq = pred_tokens[i][mask]
                true_seq = labels[i][mask]

                # Decode to text
                pred_text = self.tokenizer.decode(pred_seq, skip_special_tokens=True).strip()
                true_text = self.tokenizer.decode(true_seq, skip_special_tokens=True).strip()

                pred_texts.append(pred_text)
                label_texts.append(true_text)

        if not pred_texts:
            return {
                "bleu": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
            }

        metrics_dict = {}

        # Compute BLEU score
        if self.metrics.get('bleu') is not None:
            try:
                # BLEU expects references as list of lists
                bleu_result = self.metrics['bleu'].compute(
                    predictions=pred_texts,
                    references=[[ref] for ref in label_texts]
                )
                metrics_dict["bleu"] = float(bleu_result.get("bleu", 0.0))
            except Exception as e:
                logger.warning(f"Error computing BLEU: {e}")
                metrics_dict["bleu"] = 0.0

        # Compute ROUGE scores
        if self.metrics.get('rouge') is not None:
            try:
                rouge_result = self.metrics['rouge'].compute(
                    predictions=pred_texts,
                    references=label_texts
                )
                # Extract ROUGE-1, ROUGE-2, and ROUGE-L scores
                metrics_dict["rouge1"] = float(rouge_result.get("rouge1", 0.0))
                metrics_dict["rouge2"] = float(rouge_result.get("rouge2", 0.0))
                metrics_dict["rougeL"] = float(rouge_result.get("rougeL", 0.0))
            except Exception as e:
                logger.warning(f"Error computing ROUGE: {e}")
                metrics_dict["rouge1"] = 0.0
                metrics_dict["rouge2"] = 0.0
                metrics_dict["rougeL"] = 0.0

        return metrics_dict

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
