"""
Finetuning module for SmolLM2 using LORA PEFT
This module provides functionality to:
1. Split generated synthetic data into train/validation sets
2. Finetune SmolLM2 using LoRA (Low-Rank Adaptation)
3. Save and evaluate the finetuned model
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from datasets import Dataset as HFDataset
import evaluate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpochMetricsCallback(TrainerCallback):
    """Custom callback to display classification metrics at each epoch."""
    
    def __init__(self, label_mapping=None):
        self.label_mapping = label_mapping or {}
        self.epoch_metrics = []
        
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Called after evaluation at the end of each epoch."""
        if logs is None:
            return
            
        # Extract current epoch
        current_epoch = int(state.epoch) if state.epoch is not None else len(self.epoch_metrics) + 1
        
        # Get metrics
        eval_loss = logs.get('eval_loss', 0.0)
        eval_accuracy = logs.get('eval_accuracy', 0.0)
        eval_precision = logs.get('eval_precision', 0.0)
        eval_recall = logs.get('eval_recall', 0.0)
        eval_f1 = logs.get('eval_f1', 0.0)
        
        # Store metrics
        epoch_metric = {
            'epoch': current_epoch,
            'eval_loss': eval_loss,
            'eval_accuracy': eval_accuracy,
            'eval_precision': eval_precision,
            'eval_recall': eval_recall,
            'eval_f1': eval_f1
        }
        self.epoch_metrics.append(epoch_metric)
        
        # Display metrics
        print(f"\n📊 EPOCH {current_epoch} RESULTS:")
        print(f"   {'Metric':<12} {'Value':<8} {'Progress':<20}")
        print(f"   {'-'*12} {'-'*8} {'-'*20}")
        
        metrics_display = [
            ('Loss', eval_loss, 'lower_better'),
            ('Accuracy', eval_accuracy, 'higher_better'),
            ('Precision', eval_precision, 'higher_better'), 
            ('Recall', eval_recall, 'higher_better'),
            ('F1 Score', eval_f1, 'higher_better')
        ]
        
        for metric_name, value, direction in metrics_display:
            # Create progress bar
            if direction == 'higher_better':
                # For metrics where higher is better (0-1 range)
                bar_length = int(value * 20)
                progress_bar = '█' * bar_length + '░' * (20 - bar_length)
                emoji = '📈' if value > 0.7 else '📊' if value > 0.5 else '📉'
            else:
                # For loss where lower is better (approximate 0-2 range)
                normalized_loss = max(0, min(1, 1 - (value / 2)))  # Invert and normalize
                bar_length = int(normalized_loss * 20)
                progress_bar = '█' * bar_length + '░' * (20 - bar_length)
                emoji = '📉' if value < 0.5 else '📊' if value < 1.0 else '📈'
            
            print(f"   {emoji} {metric_name:<10} {value:<8.4f} {progress_bar}")
        
        # Show best metrics so far
        if len(self.epoch_metrics) > 1:
            best_f1 = max(m['eval_f1'] for m in self.epoch_metrics)
            best_accuracy = max(m['eval_accuracy'] for m in self.epoch_metrics)
            
            print(f"\n   🏆 Best so far:")
            print(f"      F1 Score: {best_f1:.4f}")
            print(f"      Accuracy: {best_accuracy:.4f}")
        
        print("-" * 45)
    
    def on_train_end(self, args, state, control, logs=None, **kwargs):
        """Called at the end of training to show summary."""
        if not self.epoch_metrics:
            return
            
        print(f"\n🎯 TRAINING SUMMARY:")
        print(f"   Total epochs completed: {len(self.epoch_metrics)}")
        
        # Find best epoch
        best_epoch_idx = max(range(len(self.epoch_metrics)), 
                           key=lambda i: self.epoch_metrics[i]['eval_f1'])
        best_metrics = self.epoch_metrics[best_epoch_idx]
        
        print(f"   🏆 Best performance at epoch {best_metrics['epoch']}:")
        print(f"      Accuracy:  {best_metrics['eval_accuracy']:.4f}")
        print(f"      Precision: {best_metrics['eval_precision']:.4f}")
        print(f"      Recall:    {best_metrics['eval_recall']:.4f}")
        print(f"      F1 Score:  {best_metrics['eval_f1']:.4f}")
        print(f"      Loss:      {best_metrics['eval_loss']:.4f}")
        
        # Show improvement over training
        if len(self.epoch_metrics) > 1:
            first_f1 = self.epoch_metrics[0]['eval_f1']
            final_f1 = self.epoch_metrics[-1]['eval_f1']
            improvement = final_f1 - first_f1
            
            improvement_emoji = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            print(f"   {improvement_emoji} F1 improvement: {improvement:+.4f}")


class SyntheticDataset(Dataset):
    """Custom Dataset class for synthetic text classification data."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class SmolLM2FineTuner:
    """
    Fine-tuner for SmolLM2 using LoRA PEFT for text classification tasks.
    """
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        max_length: int = 512,
        output_dir: str = "./finetuned_models"
    ):
        """
        Initialize the fine-tuner.
        
        Args:
            model_name: The base model to finetune
            max_length: Maximum sequence length for tokenization
            output_dir: Directory to save finetuned models
        """
        self.model_name = model_name
        self.max_length = max_length
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.num_labels = None
        self.label_mapping = None
        
        # Load HF token if available
        self._load_hf_token()
        
    def _load_hf_token(self):
        """Load Hugging Face token from environment."""
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if token:
            from huggingface_hub import login
            login(token)
            logger.info("✅ Logged in to Hugging Face Hub")
        else:
            logger.warning("⚠️ No HF_TOKEN found in environment")
    
    def load_and_split_data(
        self,
        data_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        text_column: str = "text",
        label_column: str = "label"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load synthetic data and split into train/validation/test sets.
        
        Args:
            data_path: Path to the CSV file containing synthetic data
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            text_column: Name of the text column
            label_column: Name of the label column
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"📂 Loading data from {data_path}")
        
        # Load data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"📊 Loaded {len(df)} samples")
        
        # Check required columns
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"Required columns {text_column}, {label_column} not found in data")
        
        # Clean data
        df = df.dropna(subset=[text_column, label_column])
        df[text_column] = df[text_column].astype(str)
        
        logger.info(f"📊 After cleaning: {len(df)} samples")
        logger.info(f"🏷️ Labels distribution:\n{df[label_column].value_counts()}")
        
        # Encode labels
        df['encoded_labels'] = self.label_encoder.fit_transform(df[label_column])
        self.num_labels = len(self.label_encoder.classes_)
        self.label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        
        logger.info(f"🎯 Number of unique labels: {self.num_labels}")
        logger.info(f"🔢 Label mapping: {self.label_mapping}")
        
        # Split data
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['encoded_labels']
        )
        
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=random_state,
            stratify=train_val_df['encoded_labels']
        )
        
        logger.info(f"📈 Data split:")
        logger.info(f"  Training: {len(train_df)} samples")
        logger.info(f"  Validation: {len(val_df)} samples") 
        logger.info(f"  Testing: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def prepare_model_and_tokenizer(self):
        """Initialize tokenizer and model with LoRA configuration."""
        logger.info(f"🤖 Loading model and tokenizer: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model for sequence classification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=16,  # Rank
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("✅ Model and tokenizer prepared with LoRA")
    
    def prepare_datasets(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        text_column: str = "text"
    ) -> Tuple[HFDataset, HFDataset]:
        """
        Prepare Hugging Face datasets for training.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            text_column: Name of the text column
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        logger.info("🔄 Preparing datasets for training")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
        
        # Convert to HF datasets
        train_dataset = HFDataset.from_pandas(train_df)
        val_dataset = HFDataset.from_pandas(val_df)
        
        # Tokenize datasets
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # Set format for PyTorch
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'encoded_labels'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'encoded_labels'])
        
        # Rename labels column
        train_dataset = train_dataset.rename_column('encoded_labels', 'labels')
        val_dataset = val_dataset.rename_column('encoded_labels', 'labels')
        
        logger.info("✅ Datasets prepared for training")
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')
        recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')
        f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
        
        return {
            'accuracy': accuracy['accuracy'],
            'precision': precision['precision'],
            'recall': recall['recall'],
            'f1': f1['f1']
        }
    
    def train(
        self,
        train_dataset: HFDataset,
        val_dataset: HFDataset,
        num_train_epochs: int = 3,
        learning_rate: float = 2e-4,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        save_strategy: str = "epoch",
        eval_strategy: str = "epoch",
        save_total_limit: int = 2,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_f1",
        greater_is_better: bool = True,
        early_stopping_patience: int = 2,
        run_name: Optional[str] = None,
        show_epoch_metrics: bool = True
    ):
        """
        Fine-tune the model using the prepared datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_train_epochs: Number of training epochs
            learning_rate: Learning rate
            per_device_train_batch_size: Training batch size per device
            per_device_eval_batch_size: Evaluation batch size per device
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay
            save_strategy: When to save checkpoints
            eval_strategy: When to evaluate
            save_total_limit: Maximum number of checkpoints to keep
            load_best_model_at_end: Whether to load best model at end
            metric_for_best_model: Metric to use for best model selection
            greater_is_better: Whether higher metric is better
            early_stopping_patience: Early stopping patience
            run_name: Name for the training run
            show_epoch_metrics: Whether to show detailed metrics at each epoch
        """
        if run_name is None:
            run_name = f"smollm2_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create output directory for this run
        run_output_dir = self.output_dir / run_name
        run_output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"🚀 Starting fine-tuning run: {run_name}")
        logger.info(f"📁 Output directory: {run_output_dir}")
        
                # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            fp16=fp16,
            gradient_checkpointing=gradient_checkpointing,
            dataloader_drop_last=False,
            eval_strategy="epoch",  # Updated parameter name
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            run_name=run_name,
            report_to="none",  # Disable wandb/tensorboard
            logging_steps=max(1, len(train_dataset) // (batch_size * 10)),
            logging_first_step=True,
            disable_tqdm=show_epoch_metrics,  # Disable default tqdm if we're showing custom metrics
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Set up callbacks
        callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        
        # Add epoch metrics callback if requested
        epoch_metrics_callback = None
        if show_epoch_metrics:
            epoch_metrics_callback = EpochMetricsCallback(label_mapping=self.label_mapping)
            callbacks.append(epoch_metrics_callback)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        # Train the model
        logger.info("🏋️ Starting training...")
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model()
        trainer.save_state()
        
        # Save label mapping
        label_mapping_path = run_output_dir / "label_mapping.json"
        with open(label_mapping_path, 'w') as f:
            json.dump(self.label_mapping, f, indent=2)
        
        # Save training metrics
        metrics_to_save = train_result.metrics.copy()
        
        # Add epoch-by-epoch metrics if available
        if epoch_metrics_callback and epoch_metrics_callback.epoch_metrics:
            metrics_to_save['epoch_metrics'] = epoch_metrics_callback.epoch_metrics
            
        metrics_path = run_output_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        logger.info("✅ Training completed!")
        logger.info(f"📊 Final training metrics: {train_result.metrics}")
        
        return trainer, train_result
    
    def evaluate_model(
        self, 
        test_df: pd.DataFrame, 
        model_path: str,
        text_column: str = "text"
    ) -> Dict:
        """
        Evaluate the fine-tuned model on test data.
        
        Args:
            test_df: Test DataFrame
            model_path: Path to the fine-tuned model
            text_column: Name of the text column
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"🧪 Evaluating model from {model_path}")
        
        # Load the fine-tuned model
        model = PeftModel.from_pretrained(
            AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            ),
            model_path
        )
        
        # Prepare test dataset
        test_dataset = HFDataset.from_pandas(test_df)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
        
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'encoded_labels'])
        test_dataset = test_dataset.rename_column('encoded_labels', 'labels')
        
        # Create trainer for evaluation
        trainer = Trainer(
            model=model,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self.compute_metrics
        )
        
        # Evaluate
        eval_results = trainer.evaluate(test_dataset)
        
        logger.info(f"📊 Test evaluation results: {eval_results}")
        
        return eval_results
    
    def finetune_pipeline(
        self,
        data_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        num_train_epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 8,
        run_name: Optional[str] = None,
        text_column: str = "text",
        label_column: str = "label",
        show_epoch_metrics: bool = True
    ) -> Tuple[str, Dict]:
        """
        Complete fine-tuning pipeline from data loading to evaluation.
        
        Args:
            data_path: Path to the synthetic data CSV file
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation  
            num_train_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size for training and evaluation
            run_name: Name for the training run
            text_column: Name of the text column
            label_column: Name of the label column
            show_epoch_metrics: Whether to display detailed metrics at each epoch
            
        Returns:
            Tuple of (model_path, evaluation_results)
        """
        logger.info("🔄 Starting complete fine-tuning pipeline")
        
        # Step 1: Load and split data
        train_df, val_df, test_df = self.load_and_split_data(
            data_path=data_path,
            test_size=test_size,
            val_size=val_size,
            text_column=text_column,
            label_column=label_column
        )
        
        # Step 2: Prepare model and tokenizer
        self.prepare_model_and_tokenizer()
        
        # Step 3: Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets(
            train_df=train_df,
            val_df=val_df,
            text_column=text_column
        )
        
        # Step 4: Train the model
        trainer, train_result = self.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            run_name=run_name,
            show_epoch_metrics=show_epoch_metrics
        )
        
        # Get model path
        model_path = trainer.args.output_dir
        
        # Step 5: Evaluate on test set
        eval_results = self.evaluate_model(
            test_df=test_df,
            model_path=model_path,
            text_column=text_column
        )
        
        logger.info("🎉 Fine-tuning pipeline completed successfully!")
        logger.info(f"📁 Model saved to: {model_path}")
        
        return model_path, eval_results


def split_synthetic_data(
    data_path: str,
    output_dir: str = "./split_data",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    text_column: str = "text",
    label_column: str = "label"
) -> Tuple[str, str, str]:
    """
    Standalone function to split synthetic data into train/val/test sets.
    
    Args:
        data_path: Path to the CSV file containing synthetic data
        output_dir: Directory to save split datasets
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_state: Random seed for reproducibility
        text_column: Name of the text column
        label_column: Name of the label column
        
    Returns:
        Tuple of paths to (train.csv, val.csv, test.csv)
    """
    logger.info(f"📊 Splitting data from {data_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Clean data
    df = df.dropna(subset=[text_column, label_column])
    df[text_column] = df[text_column].astype(str)
    
    logger.info(f"📈 Loaded {len(df)} samples")
    logger.info(f"🏷️ Labels: {df[label_column].value_counts().to_dict()}")
    
    # Encode labels for stratification
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df[label_column])
    
    # Split data
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=encoded_labels
    )
    
    # Re-encode for train/val split
    train_val_encoded = label_encoder.transform(train_val_df[label_column])
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_encoded
    )
    
    # Save splits
    train_path = output_path / "train.csv"
    val_path = output_path / "val.csv" 
    test_path = output_path / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"📁 Data split saved:")
    logger.info(f"  Train: {train_path} ({len(train_df)} samples)")
    logger.info(f"  Val: {val_path} ({len(val_df)} samples)")
    logger.info(f"  Test: {test_path} ({len(test_df)} samples)")
    
    return str(train_path), str(val_path), str(test_path)


# Example usage functions
def example_split_data():
    """Example of how to split synthetic data."""
    data_path = "../generated_data/20251107_153126.csv"
    
    train_path, val_path, test_path = split_synthetic_data(
        data_path=data_path,
        output_dir="./split_data",
        test_size=0.2,
        val_size=0.1
    )
    
    print(f"Data split completed!")
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")
    print(f"Test: {test_path}")


def example_finetune():
    """Example of how to fine-tune SmolLM2."""
    # Initialize fine-tuner
    finetuner = SmolLM2FineTuner(
        model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        output_dir="./finetuned_models"
    )
    
    # Run complete pipeline
    data_path = "../generated_data/20251107_153126.csv"
    
    model_path, eval_results = finetuner.finetune_pipeline(
        data_path=data_path,
        num_train_epochs=3,
        learning_rate=2e-4,
        batch_size=4,  # Reduce if memory issues
        run_name="polite_guard_finetune"
    )
    
    print(f"Fine-tuning completed!")
    print(f"Model saved to: {model_path}")
    print(f"Evaluation results: {eval_results}")


if __name__ == "__main__":
    # Uncomment to test splitting
    # example_split_data()
    
    # Uncomment to test fine-tuning
    # example_finetune()
    
    print("🚀 SmolLM2 Fine-tuning module ready!")
    print("Use example_split_data() to split synthetic data")
    print("Use example_finetune() to fine-tune SmolLM2 with LoRA")