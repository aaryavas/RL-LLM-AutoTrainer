"""
Data splitting utilities for train/validation/test splits.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Handles splitting of datasets into train/validation/test sets.
    Ensures stratified splits for balanced class distribution.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize the data splitter.

        Args:
            test_size: Proportion of data for testing (0-1)
            val_size: Proportion of training data for validation (0-1)
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()

    def split_dataframe(
        self,
        df: pd.DataFrame,
        label_column: str = "label",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split a dataframe into train/val/test sets.

        Args:
            df: Input dataframe
            label_column: Name of the label column

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting {len(df)} samples into train/val/test")

        # Encode labels for stratification
        encoded_labels = self.label_encoder.fit_transform(df[label_column])

        # First split: train+val vs test
        train_val_df, test_df, train_val_labels, _ = train_test_split(
            df,
            encoded_labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=encoded_labels,
        )

        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=train_val_labels,
        )

        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def split_and_save(
        self,
        data_path: str,
        output_dir: str,
        text_column: str = "text",
        label_column: str = "label",
    ) -> Tuple[str, str, str]:
        """
        Split data from CSV file and save to separate files.

        Args:
            data_path: Path to input CSV file
            output_dir: Directory to save split files
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            Tuple of paths to (train.csv, val.csv, test.csv)
        """
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples from {data_path}")

        # Clean data
        df = self._clean_dataframe(df, text_column, label_column)

        # Split data
        train_df, val_df, test_df = self.split_dataframe(df, label_column)

        # Save splits
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_path = output_path / "train.csv"
        val_path = output_path / "val.csv"
        test_path = output_path / "test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Saved splits to {output_dir}")

        return str(train_path), str(val_path), str(test_path)

    def _clean_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
    ) -> pd.DataFrame:
        """
        Clean dataframe by removing missing values and converting types.

        Args:
            df: Input dataframe
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            Cleaned dataframe
        """
        # Remove missing values
        df = df.dropna(subset=[text_column, label_column])

        # Convert text column to string
        df[text_column] = df[text_column].astype(str)

        logger.info(f"After cleaning: {len(df)} samples")
        logger.info(f"Label distribution:\n{df[label_column].value_counts()}")

        return df

    def get_label_mapping(self) -> dict:
        """
        Get mapping from encoded labels to original labels.

        Returns:
            Dictionary mapping integer labels to original labels
        """
        return {i: label for i, label in enumerate(self.label_encoder.classes_)}
