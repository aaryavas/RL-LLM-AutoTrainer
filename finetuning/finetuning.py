"""
Main VB-LoRA fine-tuning API.
Provides high-level interface for fine-tuning language models with VB-LoRA.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dotenv import load_dotenv

from .config import (
    DataConfig,
    TrainingConfig,
    VBLoRAConfig,
    OutputConfig,
    HardwareConfig,
)
from .core import (
    DataProcessor,
    TokenizerManager,
    ModelLoader,
    OptimizerFactory,
)
from .training import VBLoRATrainer
from .utils import DataSplitter, ensure_dir, save_json

logger = logging.getLogger(__name__)


class SmolLM2VBLoRAFineTuner:
    """
    Main class for fine-tuning SmolLM2 (and other LMs) with VB-LoRA.

    This class orchestrates the entire fine-tuning pipeline:
    1. Data loading and preprocessing
    2. Model and tokenizer initialization
    3. VB-LoRA adapter application
    4. Custom optimizer setup
    5. Training with callbacks
    6. Evaluation and metrics
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        output_dir: str = "./output/vblora_models",
    ):
        """
        Initialize VB-LoRA fine-tuner.

        Args:
            model_name: Model name or path from Hugging Face
            output_dir: Directory to save fine-tuned models
        """
        self.model_name = model_name
        self.output_dir = output_dir

        # Initialize configurations
        self.data_config = DataConfig()
        self.training_config = TrainingConfig()
        self.vblora_config = VBLoRAConfig()
        self.output_config = OutputConfig(output_dir=output_dir)
        self.hardware_config = HardwareConfig()

        # Initialize components (will be set up during pipeline)
        self.tokenizer_manager = None
        self.model_loader = None
        self.data_processor = None
        self.data_splitter = None

        # Load HF token
        self._load_hf_token()

        # Create output directory
        ensure_dir(self.output_dir)

        logger.info(f"Initialized VB-LoRA fine-tuner for {model_name}")

    def finetune_pipeline(
        self,
        data_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        num_train_epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        run_name: Optional[str] = None,
        text_column: str = "text",
        label_column: str = "label",
        show_epoch_metrics: bool = True,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Run complete fine-tuning pipeline.

        Args:
            data_path: Path to CSV data file
            test_size: Proportion for test set
            val_size: Proportion for validation set
            num_train_epochs: Number of training epochs
            learning_rate: Base learning rate
            batch_size: Training batch size
            run_name: Optional run name
            text_column: Name of text column in CSV
            label_column: Name of label column in CSV
            show_epoch_metrics: Whether to show detailed metrics
            **kwargs: Additional arguments for configuration

        Returns:
            Tuple of (model_path, eval_results)
        """
        logger.info("Starting VB-LoRA fine-tuning pipeline")

        # Update configurations
        self._update_configs(
            test_size=test_size,
            val_size=val_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            run_name=run_name,
            text_column=text_column,
            label_column=label_column,
            **kwargs,
        )

        # Step 1: Load and split data
        train_df, val_df, test_df = self._load_and_split_data(data_path)

        # Step 2: Setup tokenizer
        tokenizer = self._setup_tokenizer()

        # Step 3: Load and configure model
        model = self._setup_model(tokenizer)

        # Step 4: Prepare datasets
        train_dataset, eval_dataset = self._prepare_datasets(
            train_df, val_df, tokenizer
        )

        # Step 5: Setup optimizer
        optimizer = self._setup_optimizer(model)

        # Step 6: Setup trainer and train
        model_path = self._train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizer=optimizer,
            show_epoch_metrics=show_epoch_metrics,
        )

        # Step 7: Save metadata
        self._save_metadata(model_path)

        # Step 8: Evaluate (optional - depends on test set)
        eval_results = {"status": "training_complete"}

        logger.info("VB-LoRA fine-tuning pipeline completed successfully")

        return model_path, eval_results

    def _load_hf_token(self) -> None:
        """Load Hugging Face token from environment."""
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if token:
            from huggingface_hub import login
            login(token)
            logger.info("Logged in to Hugging Face Hub")
        else:
            logger.warning("No HF_TOKEN found in .env file")

    def _update_configs(self, **kwargs) -> None:
        """Update configurations from arguments."""
        # Update data config
        for key in ["test_size", "val_size", "text_column", "label_column"]:
            if key in kwargs:
                setattr(self.data_config, key, kwargs[key])

        # Update training config
        for key in ["num_train_epochs", "learning_rate"]:
            if key in kwargs:
                setattr(self.training_config, key, kwargs[key])

        # Update batch size
        if "batch_size" in kwargs:
            self.training_config.per_device_train_batch_size = kwargs["batch_size"]
            self.training_config.per_device_eval_batch_size = kwargs["batch_size"]

        # Update output config
        if "run_name" in kwargs:
            self.output_config.run_name = kwargs["run_name"]

        # Update VB-LoRA config
        for key in ["num_vectors", "lora_r"]:
            if key in kwargs:
                setattr(self.vblora_config, key, kwargs[key])

    def _load_and_split_data(self, data_path: str):
        """Load and split data."""
        logger.info("Loading and splitting data")

        self.data_splitter = DataSplitter(
            test_size=self.data_config.test_size,
            val_size=self.data_config.val_size,
            random_state=self.data_config.random_state,
        )

        # Load data
        import pandas as pd
        df = pd.read_csv(data_path)

        # Create input-output format for causal LM
        # Assuming the data has 'text' and 'label' columns
        # We'll format as instruction-response pairs
        df["input"] = df[self.data_config.text_column]
        df["output"] = df[self.data_config.label_column]

        # Split
        train_df, val_df, test_df = self.data_splitter.split_dataframe(
            df, label_column=self.data_config.label_column
        )

        return train_df, val_df, test_df

    def _setup_tokenizer(self):
        """Setup tokenizer."""
        logger.info("Setting up tokenizer")

        self.tokenizer_manager = TokenizerManager(
            model_name_or_path=self.model_name,
            use_auth_token=self.hardware_config.use_auth_token,
        )

        tokenizer = self.tokenizer_manager.load_tokenizer()

        return tokenizer

    def _setup_model(self, tokenizer):
        """Setup model with VB-LoRA."""
        logger.info("Setting up model with VB-LoRA")

        # Create model loader
        self.model_loader = ModelLoader(
            model_name_or_path=self.model_name,
            bits=self.hardware_config.bits,
            double_quant=self.hardware_config.double_quant,
            quant_type=self.hardware_config.quant_type,
            use_auth_token=self.hardware_config.use_auth_token,
            trust_remote_code=self.hardware_config.trust_remote_code,
            device_map=self.hardware_config.device_map,
            max_memory_MB=self.hardware_config.max_memory_MB,
        )

        # Load base model
        model = self.model_loader.load_model(
            use_fp16=self.training_config.fp16,
            use_bf16=self.training_config.bf16,
        )

        # Resize embeddings if needed
        self.tokenizer_manager.resize_model_embeddings(model)

        # Apply VB-LoRA
        vblora_config_dict = self.vblora_config.to_peft_config_dict()
        model = self.model_loader.apply_vblora(
            model=model,
            vblora_config=vblora_config_dict,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
        )

        return model

    def _prepare_datasets(self, train_df, val_df, tokenizer):
        """Prepare datasets."""
        logger.info("Preparing datasets")

        self.data_processor = DataProcessor(
            tokenizer=tokenizer,
            source_max_len=self.data_config.source_max_len,
            target_max_len=self.data_config.target_max_len,
        )

        train_dataset = self.data_processor.prepare_dataset(
            df=train_df,
            input_column="input",
            output_column="output",
        )

        eval_dataset = self.data_processor.prepare_dataset(
            df=val_df,
            input_column="input",
            output_column="output",
        )

        return train_dataset, eval_dataset

    def _setup_optimizer(self, model):
        """Setup optimizer with VB-LoRA parameter groups."""
        logger.info("Setting up optimizer")

        optimizer_factory = OptimizerFactory(
            learning_rate=self.training_config.learning_rate,
            learning_rate_vector_bank=self.vblora_config.learning_rate_vector_bank,
            learning_rate_logits=self.vblora_config.learning_rate_logits,
            weight_decay=self.training_config.weight_decay,
        )

        # Create mock training args for optimizer
        from transformers import TrainingArguments
        mock_args = TrainingArguments(
            output_dir=self.output_dir,
            optim=self.training_config.optim,
        )

        optimizer = optimizer_factory.create_optimizer(model, mock_args)

        return optimizer

    def _train_model(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        optimizer,
        show_epoch_metrics,
    ) -> str:
        """Train the model."""
        logger.info("Training model")

        # Get data collator
        # train_on_source=True to include input tokens in loss calculation
        # This is necessary when labels are short (e.g., category names)
        data_collator = self.data_processor.get_data_collator(train_on_source=True)

        # Create trainer
        trainer = VBLoRATrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            optimizer=optimizer,
            training_config=vars(self.training_config),
            output_config=vars(self.output_config),
        )

        # Setup and train
        trainer.setup_trainer(show_epoch_metrics=show_epoch_metrics)
        train_metrics = trainer.train()

        # Get model path
        model_path = self.output_config.output_dir

        logger.info(f"Training complete. Model saved to {model_path}")

        return model_path

    def _save_metadata(self, model_path: str) -> None:
        """Save training metadata."""
        logger.info("Saving metadata")

        metadata = {
            "model_name": self.model_name,
            "vblora_config": vars(self.vblora_config),
            "training_config": {
                "num_train_epochs": self.training_config.num_train_epochs,
                "learning_rate": self.training_config.learning_rate,
                "batch_size": self.training_config.per_device_train_batch_size,
            },
            "hardware_config": {
                "bits": self.hardware_config.bits,
                "quant_type": self.hardware_config.quant_type,
            },
        }

        metadata_path = Path(model_path) / "vblora_metadata.json"
        save_json(metadata, str(metadata_path))


def split_synthetic_data(
    data_path: str,
    output_dir: str = "./split_data",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    text_column: str = "text",
    label_column: str = "label",
) -> Tuple[str, str, str]:
    """
    Standalone function to split data into train/val/test sets.

    Args:
        data_path: Path to CSV data file
        output_dir: Output directory for splits
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        text_column: Name of text column
        label_column: Name of label column

    Returns:
        Tuple of (train_path, val_path, test_path)
    """
    logger.info(f"Splitting data from {data_path}")

    splitter = DataSplitter(
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    train_path, val_path, test_path = splitter.split_and_save(
        data_path=data_path,
        output_dir=output_dir,
        text_column=text_column,
        label_column=label_column,
    )

    logger.info("Data splitting complete")

    return train_path, val_path, test_path
