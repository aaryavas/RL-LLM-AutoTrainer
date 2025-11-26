"""
Data processor for ORPO preference data.
"""

import pandas as pd
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from datasets import Dataset

# Add parent directories to path for proper imports
_current_dir = Path(__file__).parent.resolve()
_orpo_dir = _current_dir.parent
_finetuners_dir = _orpo_dir.parent
_finetuning_dir = _finetuners_dir.parent
_prototyping_dir = _finetuning_dir.parent

if str(_prototyping_dir) not in sys.path:
    sys.path.insert(0, str(_prototyping_dir))

from finetuning.finetuners.utils.data_splitter import DataSplitter

logger = logging.getLogger(__name__)

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
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        )
        
        # We use the parent class's split_dataframe method
        train_df, val_df, test_df = splitter.split_dataframe(df, label_column="label")
        
        # Rename columns to standard names expected by ORPOTrainer if needed
        # ORPOTrainer expects: prompt, chosen, rejected (or chat template format)
        # We will standardize to: prompt, chosen, rejected
        
        def standardize(d: pd.DataFrame) -> pd.DataFrame:
            d = d.copy()
            d = d.rename(columns={
                prompt_column: "prompt",
                chosen_column: "chosen",
                rejected_column: "rejected"
            })
            return d[[ "prompt", "chosen", "rejected" ]]

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
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
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
                "rejected": rejected_response
            }
            
        # Apply formatting
        formatted_data = df.apply(format_string, axis=1).tolist()
        dataset = Dataset.from_list(formatted_data)
        
        return dataset
