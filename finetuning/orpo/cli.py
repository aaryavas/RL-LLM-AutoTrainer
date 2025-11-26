"""
CLI for ORPO training.
"""

import argparse
import logging
import sys
from finetuning.orpo.train_orpo import ORPOFineTuner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    parser = argparse.ArgumentParser(description="ORPO Fine-tuning CLI")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct", help="Base model name")
    parser.add_argument("--data_path", type=str, required=True, help="Path to preference CSV data")
    parser.add_argument("--output_dir", type=str, default="./output/orpo_models", help="Output directory")
    
    # Data args
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Column name for prompt")
    parser.add_argument("--chosen_column", type=str, default="chosen", help="Column name for chosen response")
    parser.add_argument("--rejected_column", type=str, default="rejected", help="Column name for rejected response")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test set size")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="ORPO beta parameter")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Max prompt length")
    
    args = parser.parse_args()
    
    tuner = ORPOFineTuner(
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    tuner.finetune_pipeline(
        data_path=args.data_path,
        prompt_column=args.prompt_column,
        chosen_column=args.chosen_column,
        rejected_column=args.rejected_column,
        test_size=args.test_size,
        val_size=args.val_size,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length
    )

if __name__ == "__main__":
    main()
