"""
Data processing components for VB-LoRA fine-tuning.
Handles dataset loading, tokenization, and collation.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Optional, Tuple
from datasets import Dataset as HFDataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class VBLoRADataset(Dataset):
    """
    Custom Dataset for VB-LoRA causal language modeling.
    Handles input-output pairs for instruction fine-tuning.
    """

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        tokenizer,
        source_max_len: int = 1024,
        target_max_len: int = 256,
    ):
        """
        Initialize dataset.

        Args:
            inputs: List of input texts
            outputs: List of output texts
            tokenizer: Tokenizer instance
            source_max_len: Maximum length for input sequences
            target_max_len: Maximum length for output sequences
        """
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        input_text = str(self.inputs[idx])
        output_text = str(self.outputs[idx])

        # Tokenize input and output
        source = f"{self.tokenizer.bos_token}{input_text}"
        target = f"{output_text}{self.tokenizer.eos_token}"

        source_ids = self.tokenizer(
            source,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]

        target_ids = self.tokenizer(
            target,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]

        # Combine source and target
        input_ids = source_ids + target_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "source_len": len(source_ids),
        }


class VBLoRADataCollator:
    """
    Data collator for VB-LoRA causal language modeling.
    Handles padding and label masking.
    """

    def __init__(
        self,
        tokenizer,
        train_on_source: bool = False,
    ):
        """
        Initialize collator.

        Args:
            tokenizer: Tokenizer instance
            train_on_source: Whether to compute loss on input (source) tokens
        """
        self.tokenizer = tokenizer
        self.train_on_source = train_on_source

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of instances.

        Args:
            instances: List of example dictionaries

        Returns:
            Batch dictionary with padded tensors
        """
        input_ids = [instance["input_ids"] for instance in instances]
        source_lens = [instance["source_len"] for instance in instances]

        # Pad sequences
        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        # Create labels
        labels = input_ids.clone()

        # Mask source tokens if not training on source
        if not self.train_on_source:
            for i, source_len in enumerate(source_lens):
                labels[i, :source_len] = IGNORE_INDEX

        # Create attention mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Mask padding tokens in labels
        labels[~attention_mask] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DataProcessor:
    """
    Processes data for VB-LoRA fine-tuning.
    Handles loading, cleaning, and dataset preparation.
    """

    def __init__(
        self,
        tokenizer,
        source_max_len: int = 1024,
        target_max_len: int = 256,
    ):
        """
        Initialize data processor.

        Args:
            tokenizer: Tokenizer instance
            source_max_len: Maximum length for input sequences
            target_max_len: Maximum length for output sequences
        """
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len

    def load_data_from_csv(
        self,
        data_path: str,
        text_column: str = "text",
        label_column: str = "label",
    ) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            data_path: Path to CSV file
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            Loaded and cleaned dataframe
        """
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples from {data_path}")

        # Validate columns
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in data")
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in data")

        # Clean data
        df = df.dropna(subset=[text_column, label_column])
        df[text_column] = df[text_column].astype(str)
        df[label_column] = df[label_column].astype(str)

        logger.info(f"After cleaning: {len(df)} samples")

        return df

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        input_column: str = "text",
        output_column: str = "label",
    ) -> VBLoRADataset:
        """
        Prepare PyTorch dataset from dataframe.

        Args:
            df: Input dataframe
            input_column: Column name for inputs
            output_column: Column name for outputs

        Returns:
            VBLoRADataset instance
        """
        inputs = df[input_column].tolist()
        outputs = df[output_column].tolist()

        dataset = VBLoRADataset(
            inputs=inputs,
            outputs=outputs,
            tokenizer=self.tokenizer,
            source_max_len=self.source_max_len,
            target_max_len=self.target_max_len,
        )

        logger.info(f"Prepared dataset with {len(dataset)} samples")

        return dataset

    def get_data_collator(self, train_on_source: bool = False) -> VBLoRADataCollator:
        """
        Get data collator instance.

        Args:
            train_on_source: Whether to compute loss on source tokens

        Returns:
            VBLoRADataCollator instance
        """
        return VBLoRADataCollator(
            tokenizer=self.tokenizer,
            train_on_source=train_on_source,
        )
