"""
Data processing components for VB-LoRA fine-tuning.
Handles dataset loading, tokenization, and collation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..utils import DataSplitter

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
        # Use chat template if available to match inference
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": input_text}]
            source = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
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


class VBLoRADataProcessor:
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


class PreferenceDataProcessor:
    """
    Process preference data (prompt, chosen, rejected) for ORPO training.
    """

    def __init__(
        self,
        tokenizer,
        max_prompt_length: int = 512,
        max_completion_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length

    def load_and_split_data(
        self,
        data_path: str,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
        prompt_column: str = "prompt",
        chosen_column: str = "chosen",
        rejected_column: str = "rejected",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load CSV and split into train/val/test.
        """
        logger.info(f"Loading preference data from {data_path}")
        df = pd.read_csv(data_path)

        # Ensure required columns exist
        required_cols = [prompt_column, chosen_column, rejected_column]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add dummy label for DataSplitter if needed, or just use it if it exists
        # We'll use a dummy label to leverage DataSplitter's logic (though stratification on dummy is useless, it works)
        if "label" not in df.columns:
            df["label"] = 0

        splitter = DataSplitter(
            test_size=test_size, val_size=val_size, random_state=random_state
        )

        # We use the parent class's split_dataframe method
        train_df, val_df, test_df = splitter.split_dataframe(df, label_column="label")

        # Rename columns to standard names expected by ORPOTrainer if needed
        # ORPOTrainer expects: prompt, chosen, rejected (or chat template format)
        # We will standardize to: prompt, chosen, rejected

        def standardize(d: pd.DataFrame) -> pd.DataFrame:
            d = d.copy()
            d = d.rename(
                columns={
                    prompt_column: "prompt",
                    chosen_column: "chosen",
                    rejected_column: "rejected",
                }
            )
            return d[["prompt", "chosen", "rejected"]]

        return standardize(train_df), standardize(val_df), standardize(test_df)

    def prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        """
        Convert DataFrame to Hugging Face Dataset with string format.
        """
        # Convert to string format for ORPOTrainer (avoiding chat template issues in TRL)
        # prompt: The formatted chat history (User: ...)
        # chosen: The assistant response
        # rejected: The assistant response

        def format_string(row):
            messages = [{"role": "user", "content": str(row["prompt"])}]

            # Apply chat template to get the prompt string
            # Ensure we add the generation prompt
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt_str = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # Fallback if no chat template (shouldn't happen for SmolLM2)
                prompt_str = f"User: {row['prompt']}\nAssistant:"

            # Add a space to ensure tokenization boundary, just in case
            # This helps prevent merging of the last prompt token with the first response token
            chosen_response = str(row["chosen"]).lstrip()
            rejected_response = str(row["rejected"]).lstrip()

            return {
                "prompt": prompt_str,
                "chosen": chosen_response,
                "rejected": rejected_response,
            }

        # Apply formatting
        formatted_data = df.apply(format_string, axis=1).tolist()

        # Convert list of dicts to dict of lists for HFDataset
        data_dict = {
            "prompt": [item["prompt"] for item in formatted_data],
            "chosen": [item["chosen"] for item in formatted_data],
            "rejected": [item["rejected"] for item in formatted_data],
        }
        dataset = HFDataset.from_dict(data_dict)

        return dataset
