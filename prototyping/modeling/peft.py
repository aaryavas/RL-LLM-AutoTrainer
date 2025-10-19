"""
PEFT fine-tuning framework for iterative model training.
"""
import os
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    DataCollatorForLanguageModeling,
    default_data_collator,
)
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import wandb
from accelerate import Acceleratorz

from .config import FrameworkConfig


class PEFTFineTuner:
    """Main class for PEFT fine-tuning with iterative capabilities."""

    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.accelerator = Accelerator()
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.is_initialized = False

    def initialize_model_and_tokenizer(self) -> None:
        """Initialize the base model and tokenizer."""
        if self.is_initialized:
            return

        print(f"Loading model: {self.config.model.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            trust_remote_code=self.config.model.trust_remote_code,
        )

        # Add padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate settings
        model_kwargs = {
            "device_map": self.config.model.device_map,
            "trust_remote_code": self.config.model.trust_remote_code,
            "use_cache": self.config.model.use_cache,
        }

        if self.config.model.torch_dtype:
            model_kwargs["torch_dtype"] = getattr(torch, self.config.model.torch_dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            **model_kwargs
        )

        print("Model and tokenizer loaded successfully.")
        self.is_initialized = True

    def setup_peft_config(self) -> LoraConfig:
        """Create PEFT configuration."""
        self.peft_config = LoraConfig(
            r=self.config.peft.r,
            lora_alpha=self.config.peft.lora_alpha,
            lora_dropout=self.config.peft.lora_dropout,
            bias=self.config.peft.bias,
            task_type=self.config.peft.task_type,
            target_modules=self.config.peft.target_modules,
            modules_to_save=self.config.peft.modules_to_save,
        )
        return self.peft_config

    def prepare_model_for_training(self) -> None:
        """Apply PEFT configuration to the model."""
        if not self.is_initialized:
            self.initialize_model_and_tokenizer()

        if self.peft_config is None:
            self.setup_peft_config()

        print("Applying PEFT configuration...")
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()

    def load_and_prepare_data(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Load and preprocess the dataset."""
        print(f"Loading dataset: {self.config.data.dataset_name}")

        try:
            # Try loading from Hugging Face Hub
            full_dataset = load_dataset(self.config.data.dataset_name, split=self.config.data.train_split)
        except Exception as e:
            print(f"Failed to load dataset from Hub: {e}")
            raise ValueError(f"Could not load dataset: {self.config.data.dataset_name}")

        # Preprocess function
        def preprocess_function(examples):
            return self.tokenizer(
                examples[self.config.data.text_column],
                truncation=True,
                padding=False,
                max_length=self.config.data.max_length,
            )

        # Tokenize dataset
        tokenized_dataset = full_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=self.config.data.preprocessing_num_workers,
            remove_columns=full_dataset.column_names
        )

        # Split if eval_split doesn't exist
        if self.config.data.eval_split in full_dataset.info.splits:
            try:
                eval_dataset_full = load_dataset(
                    self.config.data.dataset_name,
                    split=self.config.data.eval_split
                )
                eval_dataset = eval_dataset_full.map(
                    preprocess_function,
                    batched=True,
                    num_proc=self.config.data.preprocessing_num_workers,
                    remove_columns=eval_dataset_full.column_names
                )
            except:
                print("Using train split for evaluation")
                eval_dataset = None
        else:
            # Create train/validation split
            dataset_dict = tokenized_dataset.train_test_split(
                test_size=self.config.data.test_size,
                seed=42
            )
            tokenized_dataset = dataset_dict["train"]
            eval_dataset = dataset_dict["test"]

        self.train_dataset = tokenized_dataset
        self.eval_dataset = eval_dataset

        print(f"Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            print(",d")
        else:
            print("No evaluation dataset")

        return self.train_dataset, self.eval_dataset

    def setup_trainer(self) -> Trainer:
        """Set up the HuggingFace Trainer."""
        if not self.model:
            self.prepare_model_for_training()

        if not self.train_dataset:
            self.load_and_prepare_data()

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

        # Training arguments
        training_args = self.config.to_training_arguments()

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )

        return self.trainer

    def train_iteration(self, iteration: int = 1) -> Dict[str, Any]:
        """Run a single training iteration."""
        print(f"\n{'='*50}")
        print(f"Starting Iteration {iteration}")
        print(f"{'='*50}")

        # Update config for this iteration
        self.config.update_for_iteration(iteration)

        # Set up trainer if not done
        if not self.trainer:
            self.setup_trainer()

        # Resume from checkpoint if enabled
        resume_from_checkpoint = None
        if self.config.iterative.enable_checkpoint_resume:
            checkpoint_dir = Path(self.config.training.output_dir) / f"checkpoint-latest"
            if checkpoint_dir.exists():
                resume_from_checkpoint = str(checkpoint_dir)
                print(f"Resuming from checkpoint: {resume_from_checkpoint}")

        # Train
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Evaluate
        eval_results = {}
        if self.eval_dataset:
            eval_results = self.trainer.evaluate()
            print(f"Iteration {iteration} Eval Results: {eval_results}")

        # Save model
        self.save_checkpoint(iteration)

        # Compile results
        results = {
            "iteration": iteration,
            "train_loss": train_result.training_loss,
            "eval_results": eval_results,
            "checkpoint_path": self.config.training.output_dir,
        }

        # Log to wandb if enabled
        if self.config.iterative.log_iteration_metrics:
            self._log_iteration_metrics(iteration, results)

        return results

    def iterative_train(self) -> List[Dict[str, Any]]:
        """Run multiple iterations of training."""
        results = []

        for iteration in range(1, self.config.iterative.max_iterations + 1):
            iteration_result = self.train_iteration(iteration)
            results.append(iteration_result)

            # Check for early stopping or other conditions
            if self._should_stop_iteration(iteration, iteration_result):
                break

        # Save final combined results
        self._save_iterative_results(results)

        return results

    def save_checkpoint(self, iteration: int) -> None:
        """Save model checkpoint."""
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PEFT model
        self.model.save_pretrained(str(output_dir))

        # Save tokenizer
        self.tokenizer.save_pretrained(str(output_dir))

        # Save config
        config_path = output_dir / "peft_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "model_name": self.config.model.model_name,
                "iteration": iteration,
                "peft_config": {k: str(v) for k, v in self.peft_config.__dict__.items()},
                "training_config": {k: str(v) for k, v in self.config.training.__dict__.items()},
            }, f, indent=2)

        print(f"Checkpoint saved to: {output_dir}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        if not self.is_initialized:
            self.initialize_model_and_tokenizer()

        # Load PEFT config
        config_path = Path(checkpoint_path) / "peft_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            # Update config if needed
            self.config.model.model_name = config_data.get("model_name", self.config.model.model_name)

        # Load PEFT model
        self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        print(f"Checkpoint loaded from: {checkpoint_path}")

    def _log_iteration_metrics(self, iteration: int, results: Dict[str, Any]) -> None:
        """Log metrics for the iteration."""
        try:
            wandb.init(project="peft-finetune", name=self.config.get_experiment_name())
            wandb.log({
                f"iteration_{iteration}_train_loss": results["train_loss"],
                **{f"iteration_{iteration}_{k}": v for k, v in results.get("eval_results", {}).items()}
            })
        except ImportError:
            print("wandb not available, skipping logging")

    def _should_stop_iteration(self, iteration: int, results: Dict[str, Any]) -> bool:
        """Check if training should stop early."""
        # Implement custom stopping criteria here
        return False

    def _save_iterative_results(self, results: List[Dict[str, Any]]) -> None:
        """Save combined results from all iterations."""
        results_dict = {
            "experiment_name": self.config.get_experiment_name(),
            "iterations": results,
            "final_model_path": self.config.training.output_dir if results else None
        }

        results_file = Path(self.config.training.output_dir) / "iterative_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        print(f"Iterative training results saved to: {results_file}")

    def generate_sample(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using the fine-tuned model."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call prepare_model_for_training() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length + len(inputs["input_ids"][0]),
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


def create_default_config() -> FrameworkConfig:
    """Create a default configuration for SmolLM PEFT fine-tuning."""
    return FrameworkConfig()


def quick_train(config: Optional[FrameworkConfig] = None) -> Dict[str, Any]:
    """Quick training function with default or custom config."""
    if config is None:
        config = create_default_config()

    tuner = PEFTFineTuner(config)
    results = tuner.iterative_train()

    return {
        "results": results,
        "model_path": config.training.output_dir,
        "config": config
    }


if __name__ == "__main__":
    # Example usage
    config = create_default_config()
    tuner = PEFTFineTuner(config)

    # Run training
    results = tuner.iterative_train()

    # Test generation
    sample_prompt = "Once upon a time"
    generated = tuner.generate_sample(sample_prompt, max_length=50)
    print(f"Generated: {generated}")
