import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ORPODataGenerator:
    """
    Handles creation of ORPO datasets.
    Chosen responses are taken from the ground truth.
    Rejected responses are provided from an external source (e.g., prior generation).
    """
    
    def __init__(self):
        # No model/tokenizer needed if we are just formatting data
        pass

    def create_orpo_dataset(
        self,
        df: pd.DataFrame,
        rejected_df: pd.DataFrame,
        prompt_col: str = "text",
        chosen_col: str = "label",
        rejected_col: str = "rejected_response"
    ) -> pd.DataFrame:
        """
        Creates the final ORPO dataset by combining chosen (ground truth) and rejected responses.

        Args:
            df: DataFrame containing the prompt and chosen response (ground truth).
            rejected_df: DataFrame containing the rejected responses. Assumed to be aligned with df.
            prompt_col: Name of the column containing the input prompt in df.
            chosen_col: Name of the column containing the chosen response in df.
            rejected_col: Name of the column containing the rejected response in rejected_df.

        Returns:
            DataFrame with 'prompt', 'chosen', and 'rejected' columns ready for ORPO training.
        """
        logger.info("Creating final ORPO dataset...")

        if len(df) != len(rejected_df):
            logger.warning(f"Input dataframe length ({len(df)}) does not match rejected dataframe length ({len(rejected_df)}). Truncating to minimum length.")
            min_len = min(len(df), len(rejected_df))
            df = df.iloc[:min_len]
            rejected_df = rejected_df.iloc[:min_len]

        orpo_data = []
        
        # Reset indices to ensure alignment if they are not aligned
        df = df.reset_index(drop=True)
        rejected_df = rejected_df.reset_index(drop=True)
        
        for idx in range(len(df)):
            prompt = df.loc[idx, prompt_col]
            chosen = df.loc[idx, chosen_col]
            rejected = rejected_df.loc[idx, rejected_col]
            
            # Basic validation
            if pd.notna(prompt) and pd.notna(chosen) and pd.notna(rejected):
                # Ensure strings
                prompt = str(prompt)
                chosen = str(chosen)
                rejected = str(rejected)
                
                if prompt and chosen and rejected:
                    orpo_data.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected
                    })
            else:
                logger.warning(f"Row {idx}: Missing data. Skipping.")
        
        return pd.DataFrame(orpo_data)

