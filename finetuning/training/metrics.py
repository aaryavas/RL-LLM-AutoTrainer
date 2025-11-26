"""
Metrics computation for VB-LoRA training.
Enhanced with BLEU, ROUGE, BERTScore, and CodeBERTScore metrics for text generation quality.
"""

import numpy as np
import pandas as pd
import evaluate
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsComputer:
    """
    Computes evaluation metrics for training.
    Uses Hugging Face evaluate library for standard metrics.
    Includes BLEU, ROUGE, BERTScore, and CodeBERTScore metrics for text generation quality assessment.
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

        try:
            # Load BERTScore metric for semantic similarity
            self.metrics['bertscore'] = evaluate.load('bertscore')
            logger.info("BERTScore metric loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load BERTScore metric: {e}")
            self.metrics['bertscore'] = None

    def compute_metrics(self, eval_pred: Tuple) -> Dict[str, float]:
        """
        Compute metrics from evaluation predictions.
        Includes token accuracy, BLEU, ROUGE, BERTScore, and CodeBERTScore.

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
        Compute BLEU, ROUGE, BERTScore, and CodeBERTScore metrics for text generation quality.

        Args:
            predictions: Model predictions (logits)
            labels: Ground truth labels

        Returns:
            Dictionary with BLEU, ROUGE, BERTScore, and CodeBERTScore metrics
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
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
                "codebertscore_precision": 0.0,
                "codebertscore_recall": 0.0,
                "codebertscore_f1": 0.0,
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

        # Compute BERTScore
        if self.metrics.get('bertscore') is not None:
            try:
                bertscore_result = self.metrics['bertscore'].compute(
                    predictions=pred_texts,
                    references=label_texts,
                    lang="en"
                )
                metrics_dict["bertscore_precision"] = float(np.mean(bertscore_result["precision"]))
                metrics_dict["bertscore_recall"] = float(np.mean(bertscore_result["recall"]))
                metrics_dict["bertscore_f1"] = float(np.mean(bertscore_result["f1"]))

                # Compute CodeBERTScore (using microsoft/codebert-base)
                codebertscore_result = self.metrics['bertscore'].compute(
                    predictions=pred_texts,
                    references=label_texts,
                    model_type="microsoft/codebert-base",
                    num_layers=12,
                    rescale_with_baseline=False
                )
                metrics_dict["codebertscore_precision"] = float(np.mean(codebertscore_result["precision"]))
                metrics_dict["codebertscore_recall"] = float(np.mean(codebertscore_result["recall"]))
                metrics_dict["codebertscore_f1"] = float(np.mean(codebertscore_result["f1"]))
            except Exception as e:
                logger.warning(f"Error computing BERTScore/CodeBERTScore: {e}")
                metrics_dict["bertscore_precision"] = 0.0
                metrics_dict["bertscore_recall"] = 0.0
                metrics_dict["bertscore_f1"] = 0.0
                metrics_dict["codebertscore_precision"] = 0.0
                metrics_dict["codebertscore_recall"] = 0.0
                metrics_dict["codebertscore_f1"] = 0.0

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

    def identify_orpo_candidates(
        self,
        df: pd.DataFrame,
        prediction_col: str = "prediction",
        reference_col: str = "label",
        threshold: float = 0.82
    ) -> pd.DataFrame:
        """
        Identifies poor performing examples to be used for DPO training.

        Args:
            df: Pandas DataFrame containing predictions and references
            prediction_col: Name of the column containing model predictions
            reference_col: Name of the column containing ground truth labels
            threshold: Score threshold below which examples are considered "failed"

        Returns:
            DataFrame containing the failed examples
        """
        # Ensure BERTScore metric is loaded
        if self.metrics.get('bertscore') is None:
            try:
                self.metrics['bertscore'] = evaluate.load('bertscore')
            except Exception as e:
                logger.error(f"Could not load BERTScore metric: {e}")
                return pd.DataFrame()

        # Check/Calculate BERTScore
        if 'bertscore_f1' not in df.columns:
            logger.info("Calculating BERTScore for dataframe...")
            try:
                results = self.metrics['bertscore'].compute(
                    predictions=df[prediction_col].tolist(),
                    references=df[reference_col].tolist(),
                    lang="en"
                )
                df['bertscore_f1'] = results['f1']
            except Exception as e:
                logger.error(f"Error calculating BERTScore: {e}")
                df['bertscore_f1'] = 0.0

        # Check/Calculate CodeBERTScore
        if 'codebertscore_f1' not in df.columns:
            logger.info("Calculating CodeBERTScore for dataframe...")
            try:
                results = self.metrics['bertscore'].compute(
                    predictions=df[prediction_col].tolist(),
                    references=df[reference_col].tolist(),
                    model_type="microsoft/codebert-base",
                    num_layers=12,
                    rescale_with_baseline=False
                )
                df['codebertscore_f1'] = results['f1']
            except Exception as e:
                logger.error(f"Error calculating CodeBERTScore: {e}")
                df['codebertscore_f1'] = 0.0

        # Determine baseline metric
        # We take the max of the averages to see which metric the model is generally performing "better" on,
        # or which metric is more applicable (usually the higher one indicates better alignment)
        avg_bert = df['bertscore_f1'].mean()
        avg_code = df['codebertscore_f1'].mean()

        target_metric = 'bertscore_f1' if avg_bert > avg_code else 'codebertscore_f1'
        logger.info(f"Selected baseline metric: {target_metric} (BERT: {avg_bert:.4f}, CodeBERT: {avg_code:.4f})")

        # Filter rows below threshold
        failed_df = df[df[target_metric] < threshold].copy()

        # Fallback: If no values meet the threshold, take the lowest 10%
        if failed_df.empty and not df.empty:
            logger.info(f"No examples found below threshold {threshold}. Using bottom 10%.")
            cutoff = df[target_metric].quantile(0.10)
            failed_df = df[df[target_metric] <= cutoff].copy()

        return failed_df
