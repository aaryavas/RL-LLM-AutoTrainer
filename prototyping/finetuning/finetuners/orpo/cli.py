#!/usr/bin/env python3
"""
CLI for ORPO training.

This CLI can be used standalone or as part of the VB-LoRA -> ORPO pipeline.
It supports training on preference data with prompt/chosen/rejected format.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Add parent directories to path for proper imports
_current_dir = Path(__file__).parent.resolve()
_finetuners_dir = _current_dir.parent  # finetuners/
_finetuning_dir = _finetuners_dir.parent  # finetuning/
_prototyping_dir = _finetuning_dir.parent  # prototyping/

if str(_prototyping_dir) not in sys.path:
    sys.path.insert(0, str(_prototyping_dir))


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def validate_positive_float(value: str) -> float:
    """Validate that the input is a positive float."""
    try:
        float_value = float(value)
        if float_value <= 0:
            raise argparse.ArgumentTypeError(f"Value must be positive, got {float_value}")
        return float_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value}")


def validate_positive_int(value: str) -> int:
    """Validate that the input is a positive integer."""
    try:
        int_value = int(value)
        if int_value <= 0:
            raise argparse.ArgumentTypeError(f"Value must be positive, got {int_value}")
        return int_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value}")


def validate_proportion(value: str) -> float:
    """Validate that the input is a proportion between 0 and 1."""
    try:
        float_value = float(value)
        if not 0 < float_value < 1:
            raise argparse.ArgumentTypeError(f"Proportion must be between 0 and 1, got {float_value}")
        return float_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid proportion value: {value}")


def setup_argparser() -> argparse.ArgumentParser:
    """Set up the argument parser for ORPO CLI."""
    parser = argparse.ArgumentParser(
        description="ORPO (Odds Ratio Preference Optimization) Fine-tuning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic ORPO training
  python cli.py --data_path preference_data.csv

  # Use merged VB-LoRA model
  python cli.py --data_path preference_data.csv --model_name ./merged_model

  # Custom training parameters
  python cli.py --data_path preference_data.csv --epochs 5 --lr 5e-6 --beta 0.5

  # Dry run to check configuration
  python cli.py --data_path preference_data.csv --dry-run
        """
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model configuration')
    model_group.add_argument(
        "--model_name", "--model", type=str, 
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        help="Base model name or path to merged model"
    )
    model_group.add_argument(
        "--output_dir", type=str, 
        default="./output/orpo_models",
        help="Output directory for trained model"
    )
    
    # Data configuration
    data_group = parser.add_argument_group('Data configuration')
    data_group.add_argument(
        "--data_path", type=str, required=True,
        help="Path to preference CSV data (must have prompt, chosen, rejected columns)"
    )
    data_group.add_argument(
        "--prompt_column", type=str, default="prompt",
        help="Column name for prompts"
    )
    data_group.add_argument(
        "--chosen_column", type=str, default="chosen",
        help="Column name for chosen/preferred responses"
    )
    data_group.add_argument(
        "--rejected_column", type=str, default="rejected",
        help="Column name for rejected responses"
    )
    data_group.add_argument(
        "--test_size", type=validate_proportion, default=0.1,
        help="Proportion of data for testing (default: 0.1)"
    )
    data_group.add_argument(
        "--val_size", type=validate_proportion, default=0.1,
        help="Proportion of data for validation (default: 0.1)"
    )
    
    # Training configuration
    train_group = parser.add_argument_group('Training configuration')
    train_group.add_argument(
        "--epochs", type=validate_positive_int, default=3,
        help="Number of training epochs (default: 3)"
    )
    train_group.add_argument(
        "--batch_size", type=validate_positive_int, default=4,
        help="Batch size per device (default: 4)"
    )
    train_group.add_argument(
        "--lr", "--learning_rate", type=validate_positive_float, default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    train_group.add_argument(
        "--gradient_accumulation", type=validate_positive_int, default=1,
        help="Gradient accumulation steps (default: 1)"
    )
    
    # ORPO-specific configuration
    orpo_group = parser.add_argument_group('ORPO configuration')
    orpo_group.add_argument(
        "--beta", type=validate_positive_float, default=0.1,
        help="ORPO beta parameter - controls preference strength (default: 0.1)"
    )
    orpo_group.add_argument(
        "--max_prompt_length", type=validate_positive_int, default=512,
        help="Maximum prompt length (default: 512)"
    )
    orpo_group.add_argument(
        "--max_completion_length", type=validate_positive_int, default=1024,
        help="Maximum completion length (default: 1024)"
    )
    
    # Hardware configuration
    hw_group = parser.add_argument_group('Hardware configuration')
    hw_group.add_argument(
        "--bits", type=int, choices=[4, 8, 16, 32], default=4,
        help="Quantization bits (default: 4)"
    )
    hw_group.add_argument(
        "--bf16", action="store_true",
        help="Use bfloat16 precision"
    )
    hw_group.add_argument(
        "--fp16", action="store_true",
        help="Use float16 precision"
    )
    
    # Output configuration
    output_group = parser.add_argument_group('Output configuration')
    output_group.add_argument(
        "--run_name", type=str, default=None,
        help="Name for this training run (default: auto-generated)"
    )
    output_group.add_argument(
        "--save_steps", type=validate_positive_int, default=500,
        help="Save checkpoint every N steps (default: 500)"
    )
    
    # Misc configuration
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    misc_group.add_argument(
        "--dry-run", action="store_true",
        help="Show configuration without training"
    )
    misc_group.add_argument(
        "--no-metrics", action="store_true",
        help="Disable detailed metrics computation"
    )
    
    return parser


def main():
    """Main entry point for ORPO CLI."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate data file exists
    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data file not found: {args.data_path}")
        sys.exit(1)
    
    print("🚀 Starting ORPO Fine-tuning")
    print(f"📂 Data: {args.data_path}")
    print(f"🤖 Model: {args.model_name}")
    
    # Build configuration display
    config = {
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "data_path": args.data_path,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "beta": args.beta,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "bits": args.bits,
        "bf16": args.bf16,
        "fp16": args.fp16,
    }
    
    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    if args.dry_run:
        print("\n🔍 Dry run mode - configuration shown above")
        return
    
    # Import here to avoid slow imports for --help
    from finetuning.finetuners.orpo.train_orpo import ORPOFineTuner
    
    try:
        # Initialize ORPO trainer
        tuner = ORPOFineTuner(
            model_name=args.model_name,
            output_dir=args.output_dir
        )
        
        # Update hardware config
        tuner.hardware_config.bits = args.bits
        tuner.training_config.bf16 = args.bf16
        tuner.training_config.fp16 = args.fp16
        tuner.training_config.gradient_accumulation_steps = args.gradient_accumulation
        
        # Run training
        output_path = tuner.finetune_pipeline(
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
            max_prompt_length=args.max_prompt_length,
            run_name=args.run_name,
        )
        
        print(f"\n🎉 ORPO training completed successfully!")
        print(f"📁 Model saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ ORPO training failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
