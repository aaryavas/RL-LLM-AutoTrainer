"""
Tokenizer management for VB-LoRA fine-tuning.
"""

from transformers import AutoTokenizer
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DEFAULT_PAD_TOKEN = "[PAD]"


class TokenizerManager:
    """
    Manages tokenizer loading and configuration for VB-LoRA fine-tuning.
    Ensures proper special token setup.
    """

    def __init__(self, model_name_or_path: str, use_auth_token: bool = False):
        """
        Initialize tokenizer manager.

        Args:
            model_name_or_path: Model name or path
            use_auth_token: Whether to use HuggingFace auth token
        """
        self.model_name_or_path = model_name_or_path
        self.use_auth_token = use_auth_token
        self.tokenizer = None

    def load_tokenizer(self) -> AutoTokenizer:
        """
        Load and configure tokenizer.

        Returns:
            Configured tokenizer instance
        """
        logger.info(f"Loading tokenizer from {self.model_name_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            padding_side="right",
            use_fast=False,
            trust_remote_code=False,
            use_auth_token=self.use_auth_token,
        )

        # Configure special tokens
        self._configure_special_tokens()

        logger.info("Tokenizer loaded and configured successfully")

        return self.tokenizer

    def _configure_special_tokens(self) -> None:
        """Configure special tokens (pad, bos, eos, unk)."""
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self._add_pad_token()

        # Ensure BOS and EOS tokens are set
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

        logger.info(f"Special tokens configured:")
        logger.info(f"  PAD: {self.tokenizer.pad_token}")
        logger.info(f"  BOS: {self.tokenizer.bos_token}")
        logger.info(f"  EOS: {self.tokenizer.eos_token}")

    def _add_pad_token(self) -> None:
        """Add padding token to tokenizer."""
        # Try to use EOS token as padding token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Using EOS token as padding token")
        else:
            # Add a new padding token
            self.tokenizer.add_special_tokens(
                {"pad_token": DEFAULT_PAD_TOKEN}
            )
            logger.info(f"Added new padding token: {DEFAULT_PAD_TOKEN}")

    def get_vocab_size(self) -> int:
        """
        Get vocabulary size.

        Returns:
            Size of tokenizer vocabulary
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded yet")
        return len(self.tokenizer)

    def resize_model_embeddings(self, model) -> None:
        """
        Resize model embeddings to match tokenizer vocabulary.

        Args:
            model: Model to resize
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded yet")

        vocab_size = len(self.tokenizer)
        current_size = model.get_input_embeddings().weight.shape[0]

        if vocab_size != current_size:
            logger.info(f"Resizing embeddings from {current_size} to {vocab_size}")
            model.resize_token_embeddings(vocab_size)

            # Initialize new embeddings
            self._initialize_new_embeddings(model, current_size, vocab_size)

    def _initialize_new_embeddings(
        self,
        model,
        old_size: int,
        new_size: int,
    ) -> None:
        """
        Initialize newly added token embeddings.

        Args:
            model: Model with resized embeddings
            old_size: Original vocabulary size
            new_size: New vocabulary size
        """
        if new_size <= old_size:
            return

        num_new_tokens = new_size - old_size

        # Get embeddings
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # Calculate mean of existing embeddings
        input_mean = input_embeddings[:old_size].mean(dim=0, keepdim=True)
        output_mean = output_embeddings[:old_size].mean(dim=0, keepdim=True)

        # Initialize new embeddings with mean
        input_embeddings[old_size:] = input_mean
        output_embeddings[old_size:] = output_mean

        logger.info(f"Initialized {num_new_tokens} new token embeddings")
