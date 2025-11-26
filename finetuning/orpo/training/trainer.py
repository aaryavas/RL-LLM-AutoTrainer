"""
ORPO Trainer Wrapper.
"""

import logging
from typing import Dict, Any, Optional
from trl import ORPOTrainer, ORPOConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, VBLoRAConfig, TaskType
from finetuning.config.vblora_config import VBLoRADefaults

logger = logging.getLogger(__name__)

class ORPOTrainerWrapper:
    """
    Wrapper around trl.ORPOTrainer to handle configuration and setup.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset,
        eval_dataset,
        training_config: Dict[str, Any],
        orpo_config: Dict[str, Any],
        output_config: Dict[str, Any],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_config = training_config
        self.orpo_config = orpo_config
        self.output_config = output_config
        self.trainer = None

    def setup_trainer(self):
        """
        Initialize the ORPOTrainer.
        """
        logger.info("Setting up ORPO Trainer...")

        # Merge configs into ORPOConfig
        # ORPOConfig inherits from TrainingArguments, so we can pass standard training args here too.
        
        # Calculate max_length
        max_length = self.orpo_config["max_prompt_length"] + self.orpo_config["max_completion_length"]

        args = ORPOConfig(
            output_dir=self.output_config["output_dir"],
            # ORPO specific
            beta=self.orpo_config["beta"],
            max_prompt_length=self.orpo_config["max_prompt_length"],
            max_completion_length=self.orpo_config["max_completion_length"],
            max_length=max_length,
            disable_dropout=self.orpo_config["disable_dropout"],
            
            # Standard Training Args
            num_train_epochs=self.training_config["num_train_epochs"],
            learning_rate=self.training_config["learning_rate"],
            per_device_train_batch_size=self.training_config["per_device_train_batch_size"],
            per_device_eval_batch_size=self.training_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.training_config["gradient_accumulation_steps"],
            warmup_steps=self.training_config["warmup_steps"],
            weight_decay=self.training_config["weight_decay"],
            logging_steps=self.training_config["logging_steps"],
            save_steps=self.training_config["save_steps"],
            eval_strategy=self.training_config["eval_strategy"],
            save_strategy=self.training_config["save_strategy"],
            fp16=self.training_config["fp16"],
            bf16=self.training_config["bf16"],
            gradient_checkpointing=self.training_config["gradient_checkpointing"],
            optim=self.training_config["optim"],
            report_to=self.output_config.get("report_to", "none"),
            run_name=self.output_config.get("run_name"),
            remove_unused_columns=False, # Important for custom datasets sometimes
            dataloader_num_workers=0, # Prevent deadlocks
        )

        # Define VB-LoRA Config
        # Use standard defaults from codebase
        vblora_defaults = VBLoRADefaults.standard()
        peft_config_dict = vblora_defaults.to_peft_config_dict()
        
        peft_config = VBLoRAConfig(
            **peft_config_dict,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )

        self.trainer = ORPOTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )

    def train(self) -> Dict[str, Any]:
        """
        Run training.
        """
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call setup_trainer() first.")
        
        logger.info("Starting ORPO training...")
        train_result = self.trainer.train()
        
        self.trainer.save_model()
        
        return train_result.metrics

    def get_log_history(self):
        """Return the log history from the trainer state."""
        if self.trainer:
            return self.trainer.state.log_history
        return []
