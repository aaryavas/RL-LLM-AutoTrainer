"""
Main ORPO Fine-tuning Orchestrator.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for proper imports
_current_dir = Path(__file__).parent.resolve()
_finetuners_dir = _current_dir.parent  # finetuners/
_finetuning_dir = _finetuners_dir.parent  # finetuning/
_prototyping_dir = _finetuning_dir.parent  # prototyping/

if str(_prototyping_dir) not in sys.path:
    sys.path.insert(0, str(_prototyping_dir))

from finetuning.finetuners.config.base_config import (
    DataConfig,
    TrainingConfig,
    OutputConfig,
    HardwareConfig,
)
from finetuning.finetuners.orpo.config.orpo_config import ORPOSpecificConfig
from finetuning.finetuners.core.tokenizer_manager import TokenizerManager
from finetuning.finetuners.core.model_loader import ModelLoader
from finetuning.finetuners.orpo.core.data_processor import PreferenceDataProcessor
from finetuning.finetuners.orpo.training.trainer import ORPOTrainerWrapper
from finetuning.finetuners.orpo.training.visualization import TrainingVisualizer
from finetuning.finetuners.utils.helpers import ensure_dir, save_json
from finetuning.finetuners.training.metrics import MetricsComputer

logger = logging.getLogger(__name__)

class ORPOFineTuner:
    """
    Orchestrates the ORPO fine-tuning process.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        output_dir: str = "./output/orpo_models",
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Initialize configs
        self.data_config = DataConfig()
        self.training_config = TrainingConfig()
        self.orpo_config = ORPOSpecificConfig()
        self.output_config = OutputConfig(output_dir=output_dir)
        self.hardware_config = HardwareConfig()
        
        self._load_hf_token()
        ensure_dir(self.output_dir)
        
        logger.info(f"Initialized ORPO Fine-tuner for {model_name}")

    def _load_hf_token(self) -> None:
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if token:
            from huggingface_hub import login
            login(token)
            logger.info("Logged in to Hugging Face Hub")

    def finetune_pipeline(
        self,
        data_path: str,
        prompt_column: str = "prompt",
        chosen_column: str = "chosen",
        rejected_column: str = "rejected",
        **kwargs
    ) -> str:
        """
        Run the complete ORPO pipeline.
        """
        logger.info("Starting ORPO pipeline...")
        
        # Update configs from kwargs
        self._update_configs(**kwargs)
        
        # 1. Setup Tokenizer
        tokenizer_manager = TokenizerManager(
            model_name_or_path=self.model_name,
            use_auth_token=self.hardware_config.use_auth_token
        )
        tokenizer = tokenizer_manager.load_tokenizer()
        
        # 2. Load and Split Data
        data_processor = PreferenceDataProcessor(
            tokenizer=tokenizer,
            max_prompt_length=self.orpo_config.max_prompt_length,
            max_completion_length=self.orpo_config.max_completion_length
        )
        
        train_df, val_df, test_df = data_processor.load_and_split_data(
            data_path=data_path,
            test_size=self.data_config.test_size,
            val_size=self.data_config.val_size,
            prompt_column=prompt_column,
            chosen_column=chosen_column,
            rejected_column=rejected_column
        )
        
        train_dataset = data_processor.prepare_dataset(train_df)
        eval_dataset = data_processor.prepare_dataset(val_df)
        
        # 3. Setup Model
        model_loader = ModelLoader(
            model_name_or_path=self.model_name,
            bits=self.hardware_config.bits,
            double_quant=self.hardware_config.double_quant,
            quant_type=self.hardware_config.quant_type,
            device_map=self.hardware_config.device_map,
            max_memory_MB=self.hardware_config.max_memory_MB,
            use_auth_token=self.hardware_config.use_auth_token
        )
        
        model = model_loader.load_model(
            use_fp16=self.training_config.fp16,
            use_bf16=self.training_config.bf16
        )
        
        # Resize embeddings if needed (e.g. if tokenizer added tokens)
        # Usually ORPO doesn't add tokens unless chat template does something weird, but good practice.
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))

        # 4. Setup Trainer
        trainer_wrapper = ORPOTrainerWrapper(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=vars(self.training_config),
            orpo_config=vars(self.orpo_config),
            output_config=vars(self.output_config)
        )
        
        trainer_wrapper.setup_trainer()
        
        # 5. Train
        metrics = trainer_wrapper.train()
        
        # 6. Visualize
        visualizer = TrainingVisualizer(self.output_dir)
        visualizer.plot_metrics(trainer_wrapper.get_log_history())
        
        # 7. Save Metadata
        self._save_metadata(metrics)
        
        logger.info(f"ORPO training complete. Model saved to {self.output_dir}")
        return self.output_dir

    def _update_configs(self, **kwargs):
        """Update configurations from arguments."""
        # Map kwargs to config objects
        # Data Config
        if "test_size" in kwargs: self.data_config.test_size = kwargs["test_size"]
        if "val_size" in kwargs: self.data_config.val_size = kwargs["val_size"]
        
        # Training Config
        if "num_train_epochs" in kwargs: self.training_config.num_train_epochs = kwargs["num_train_epochs"]
        if "learning_rate" in kwargs: self.training_config.learning_rate = kwargs["learning_rate"]
        if "batch_size" in kwargs:
            self.training_config.per_device_train_batch_size = kwargs["batch_size"]
            self.training_config.per_device_eval_batch_size = kwargs["batch_size"]
            
        # ORPO Config
        if "beta" in kwargs: self.orpo_config.beta = kwargs["beta"]
        if "max_prompt_length" in kwargs: self.orpo_config.max_prompt_length = kwargs["max_prompt_length"]
        
        # Output Config
        if "run_name" in kwargs: self.output_config.run_name = kwargs["run_name"]

    def _save_metadata(self, metrics: Dict[str, Any]):
        metadata = {
            "model_name": self.model_name,
            "orpo_config": vars(self.orpo_config),
            "training_config": vars(self.training_config),
            "final_metrics": metrics
        }
        save_json(metadata, os.path.join(self.output_dir, "orpo_metadata.json"))
