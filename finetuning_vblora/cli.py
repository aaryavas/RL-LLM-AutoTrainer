#!/usr/bin/env python3
"""
Command-line interface for VB-LoRA fine-tuning.
Preserves the command structure from the original LoRA implementation.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
import json
import logging

from finetuning import SmolLM2VBLoRAFineTuner, split_synthetic_data
from config import (
    SMOLLM2_VARIANTS,
    PRESET_CONFIGS,
    get_variant_config,
    get_preset_config,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolLM2 models using VB-LoRA PEFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split data only
  python cli.py split data.csv --output-dir ./split_data

  # Basic fine-tuning
  python cli.py finetune data.csv

  # Use a preset configuration
  python cli.py finetune data.csv --preset standard

  # Use a specific model variant
  python cli.py finetune data.csv --variant SmolLM2-360M

  # Advanced fine-tuning with custom parameters
  python cli.py finetune data.csv --epochs 5 --lr 1e-4 --batch-size 4 --early-stopping 3

  # High capacity VB-LoRA
  python cli.py finetune data.csv --num-vectors 2048 --lora-r 8
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Split data command
    setup_split_parser(subparsers)

    # Fine-tune command
    setup_finetune_parser(subparsers)

    return parser


def setup_split_parser(subparsers):
    """Setup split command parser."""
    split_parser = subparsers.add_parser('split', help='Split data into train/val/test')
    split_parser.add_argument('data_path', type=str, help='Path to the data CSV file')
    split_parser.add_argument('--output-dir', type=str, default='./split_data',
                            help='Output directory for split data (default: ./split_data)')
    split_parser.add_argument('--test-size', type=validate_proportion, default=0.2,
                            help='Proportion of data for testing (default: 0.2)')
    split_parser.add_argument('--val-size', type=validate_proportion, default=0.1,
                            help='Proportion of training data for validation (default: 0.1)')
    split_parser.add_argument('--random-state', type=int, default=42,
                            help='Random seed for reproducibility (default: 42)')
    split_parser.add_argument('--text-column', type=str, default='text',
                            help='Name of the text column (default: text)')
    split_parser.add_argument('--label-column', type=str, default='label',
                            help='Name of the label column (default: label)')


def setup_finetune_parser(subparsers):
    """Setup finetune command parser."""
    finetune_parser = subparsers.add_parser('finetune', help='Fine-tune with VB-LoRA')
    finetune_parser.add_argument('data_path', type=str, help='Path to the data CSV file')

    # Model selection
    model_group = finetune_parser.add_mutually_exclusive_group()
    model_group.add_argument('--model', type=str, default='HuggingFaceTB/SmolLM2-1.7B-Instruct',
                           help='Model name or path (default: HuggingFaceTB/SmolLM2-1.7B-Instruct)')
    model_group.add_argument('--variant', type=str, choices=list(SMOLLM2_VARIANTS.keys()),
                           help=f'Use a predefined model variant: {list(SMOLLM2_VARIANTS.keys())}')

    # Preset configuration
    finetune_parser.add_argument('--preset', type=str, choices=list(PRESET_CONFIGS.keys()),
                               help=f'Use a preset configuration: {list(PRESET_CONFIGS.keys())}')

    # Data splitting parameters
    data_group = finetune_parser.add_argument_group('Data splitting options')
    data_group.add_argument('--test-size', type=validate_proportion, default=0.2,
                          help='Proportion of data for testing (default: 0.2)')
    data_group.add_argument('--val-size', type=validate_proportion, default=0.1,
                          help='Proportion of training data for validation (default: 0.1)')
    data_group.add_argument('--text-column', type=str, default='text',
                          help='Name of the text column (default: text)')
    data_group.add_argument('--label-column', type=str, default='label',
                          help='Name of the label column (default: label)')

    # Training parameters
    train_group = finetune_parser.add_argument_group('Training options')
    train_group.add_argument('--epochs', type=validate_positive_int, default=3,
                           help='Number of training epochs (default: 3)')
    train_group.add_argument('--lr', '--learning-rate', type=validate_positive_float, default=2e-4,
                           help='Base learning rate (default: 2e-4)')
    train_group.add_argument('--batch-size', type=validate_positive_int, default=4,
                           help='Batch size per device (default: 4)')
    train_group.add_argument('--early-stopping', type=validate_positive_int, default=2,
                           help='Early stopping patience (default: 2)')

    # VB-LoRA specific parameters
    vblora_group = finetune_parser.add_argument_group('VB-LoRA options')
    vblora_group.add_argument('--num-vectors', type=validate_positive_int, default=90,
                             help='Number of vectors in vector bank (default: 90)')
    vblora_group.add_argument('--vector-length', type=validate_positive_int, default=160,
                             help='Length of each vector - must divide model dimensions evenly (default: 160)')
    vblora_group.add_argument('--lora-r', type=validate_positive_int, default=4,
                             help='LoRA rank (default: 4)')
    vblora_group.add_argument('--lr-vector-bank', type=validate_positive_float, default=1e-3,
                             help='Learning rate for vector bank (default: 1e-3)')
    vblora_group.add_argument('--lr-logits', type=validate_positive_float, default=1e-2,
                             help='Learning rate for logits (default: 1e-2)')

    # Output options
    output_group = finetune_parser.add_argument_group('Output options')
    output_group.add_argument('--output-dir', type=str, default='./output/vblora_models',
                            help='Output directory for fine-tuned models')
    output_group.add_argument('--run-name', type=str, default=None,
                            help='Name for this training run (default: auto-generated)')

    # Hardware options
    hw_group = finetune_parser.add_argument_group('Hardware options')
    hw_group.add_argument('--bits', type=int, choices=[4, 8, 16, 32], default=4,
                        help='Quantization bits (default: 4)')
    hw_group.add_argument('--bf16', action='store_true',
                        help='Use bfloat16 precision')
    hw_group.add_argument('--fp16', action='store_true',
                        help='Use float16 precision')

    # Miscellaneous
    misc_group = finetune_parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--verbose', action='store_true',
                          help='Enable verbose logging')
    misc_group.add_argument('--dry-run', action='store_true',
                          help='Show configuration without training')
    misc_group.add_argument('--no-epoch-metrics', action='store_true',
                          help='Disable detailed metrics display at each epoch')

    return finetune_parser


def handle_split_command(args):
    """Handle the split data command."""
    print(f"🔄 Splitting data from {args.data_path}")

    train_path, val_path, test_path = split_synthetic_data(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        text_column=args.text_column,
        label_column=args.label_column
    )

    print(f"✅ Data splitting completed!")
    print(f"📁 Files saved:")
    print(f"  Train: {train_path}")
    print(f"  Validation: {val_path}")
    print(f"  Test: {test_path}")


def handle_finetune_command(args):
    """Handle the fine-tune command."""
    print(f"🚀 Starting VB-LoRA fine-tuning")
    print(f"📂 Data: {args.data_path}")

    # Determine model name
    if args.variant:
        variant_config = get_variant_config(args.variant)
        model_name = variant_config.model_name
        print(f"🤖 Using variant: {args.variant} ({model_name})")
    else:
        model_name = args.model
        print(f"🤖 Using model: {model_name}")

    # Build configuration
    config = {
        'test_size': args.test_size,
        'val_size': args.val_size,
        'num_train_epochs': args.epochs,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'run_name': args.run_name,
        'text_column': args.text_column,
        'label_column': args.label_column,
        'num_vectors': args.num_vectors,
        'lora_r': args.lora_r,
    }

    # Apply preset if specified
    if args.preset:
        preset_config = get_preset_config(args.preset)
        config.update(preset_config)
        print(f"⚙️ Using preset: {args.preset}")

    # Apply variant-specific overrides
    if args.variant:
        variant_dict = variant_config.to_dict()
        for key in ['batch_size', 'learning_rate', 'num_vectors']:
            if key in variant_dict:
                config[key] = variant_dict[key]

    # Show configuration
    print(f"\n📋 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  bits: {args.bits}")
    print(f"  bf16: {args.bf16}")
    print(f"  num_vectors: {args.num_vectors}")
    print(f"  vector_length: {args.vector_length}")
    print(f"  lora_r: {args.lora_r}")

    if args.dry_run:
        print("\n🔍 Dry run mode - configuration shown above")
        return

    # Initialize fine-tuner
    finetuner = SmolLM2VBLoRAFineTuner(
        model_name=model_name,
        output_dir=args.output_dir
    )

    # Update VB-LoRA config
    finetuner.vblora_config.num_vectors = args.num_vectors
    finetuner.vblora_config.vector_length = args.vector_length
    finetuner.vblora_config.lora_r = args.lora_r
    finetuner.vblora_config.learning_rate_vector_bank = args.lr_vector_bank
    finetuner.vblora_config.learning_rate_logits = args.lr_logits

    # Update hardware config
    finetuner.hardware_config.bits = args.bits
    finetuner.training_config.bf16 = args.bf16
    finetuner.training_config.fp16 = args.fp16
    finetuner.training_config.early_stopping_patience = args.early_stopping

    try:
        # Run fine-tuning pipeline
        model_path, eval_results = finetuner.finetune_pipeline(
            data_path=args.data_path,
            show_epoch_metrics=not args.no_epoch_metrics,
            **config
        )

        print(f"\n🎉 Fine-tuning completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"📊 Results: {eval_results}")

    except Exception as e:
        print(f"❌ Fine-tuning failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    parser = setup_argparser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Validate input file exists
    if hasattr(args, 'data_path'):
        if not os.path.exists(args.data_path):
            print(f"❌ Error: Data file not found: {args.data_path}")
            sys.exit(1)

    # Set logging level
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Route to appropriate handler
    if args.command == 'split':
        handle_split_command(args)
    elif args.command == 'finetune':
        handle_finetune_command(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
