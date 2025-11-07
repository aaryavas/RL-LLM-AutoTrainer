#!/usr/bin/env python3
"""
Command-line interface for SmolLM2 fine-tuning.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
import json

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from finetuning import SmolLM2FineTuner, split_synthetic_data
from config import (
    MODEL_VARIANTS, PRESETS, get_config_for_variant, 
    get_preset_config, merge_configs, TRAINING_CONFIG, DATA_CONFIG
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
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolLM2 models using LoRA PEFT on synthetic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive step-by-step fine-tuning (recommended for beginners)
  python cli.py interactive
  
  # Show demo of interactive process
  python cli.py interactive --demo
  
  # Basic fine-tuning
  python cli.py finetune data.csv --epochs 3 --batch-size 8
  
  # Use a preset configuration
  python cli.py finetune data.csv --preset standard
  
  # Use a specific model variant
  python cli.py finetune data.csv --variant SmolLM2-360M
  
  # Split data only
  python cli.py split data.csv --output-dir ./split_data
  
  # Advanced fine-tuning with custom parameters
  python cli.py finetune data.csv --epochs 5 --lr 1e-4 --batch-size 4 --early-stopping 3
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Interactive fine-tuning command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive step-by-step fine-tuning')
    interactive_parser.add_argument('--demo', action='store_true',
                                  help='Show a demo of the interactive process')
    
    # Split data command
    split_parser = subparsers.add_parser('split', help='Split synthetic data into train/val/test')
    split_parser.add_argument('data_path', type=str, help='Path to the synthetic data CSV file')
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
    
    # Fine-tune command
    finetune_parser = subparsers.add_parser('finetune', help='Fine-tune SmolLM2 on synthetic data')
    finetune_parser.add_argument('data_path', type=str, help='Path to the synthetic data CSV file')
    
    # Model selection
    model_group = finetune_parser.add_mutually_exclusive_group()
    model_group.add_argument('--model', type=str, default='HuggingFaceTB/SmolLM2-1.7B-Instruct',
                           help='Model name or path (default: HuggingFaceTB/SmolLM2-1.7B-Instruct)')
    model_group.add_argument('--variant', type=str, choices=list(MODEL_VARIANTS.keys()),
                           help=f'Use a predefined model variant: {list(MODEL_VARIANTS.keys())}')
    
    # Preset configuration
    finetune_parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                               help=f'Use a preset configuration: {list(PRESETS.keys())}')
    
    # Data splitting parameters
    data_group = finetune_parser.add_argument_group('Data splitting options')
    data_group.add_argument('--test-size', type=validate_proportion, default=DATA_CONFIG['test_size'],
                          help=f'Proportion of data for testing (default: {DATA_CONFIG["test_size"]})')
    data_group.add_argument('--val-size', type=validate_proportion, default=DATA_CONFIG['val_size'],
                          help=f'Proportion of training data for validation (default: {DATA_CONFIG["val_size"]})')
    data_group.add_argument('--text-column', type=str, default=DATA_CONFIG['text_column'],
                          help=f'Name of the text column (default: {DATA_CONFIG["text_column"]})')
    data_group.add_argument('--label-column', type=str, default=DATA_CONFIG['label_column'],
                          help=f'Name of the label column (default: {DATA_CONFIG["label_column"]})')
    
    # Training parameters
    train_group = finetune_parser.add_argument_group('Training options')
    train_group.add_argument('--epochs', type=validate_positive_int, 
                           default=TRAINING_CONFIG['num_train_epochs'],
                           help=f'Number of training epochs (default: {TRAINING_CONFIG["num_train_epochs"]})')
    train_group.add_argument('--lr', '--learning-rate', type=validate_positive_float,
                           default=TRAINING_CONFIG['learning_rate'],
                           help=f'Learning rate (default: {TRAINING_CONFIG["learning_rate"]})')
    train_group.add_argument('--batch-size', type=validate_positive_int,
                           default=TRAINING_CONFIG['per_device_train_batch_size'],
                           help=f'Batch size per device (default: {TRAINING_CONFIG["per_device_train_batch_size"]})')
    train_group.add_argument('--warmup-steps', type=validate_positive_int,
                           default=TRAINING_CONFIG['warmup_steps'],
                           help=f'Number of warmup steps (default: {TRAINING_CONFIG["warmup_steps"]})')
    train_group.add_argument('--weight-decay', type=validate_positive_float,
                           default=TRAINING_CONFIG['weight_decay'],
                           help=f'Weight decay (default: {TRAINING_CONFIG["weight_decay"]})')
    train_group.add_argument('--early-stopping', type=validate_positive_int,
                           default=TRAINING_CONFIG['early_stopping_patience'],
                           help=f'Early stopping patience (default: {TRAINING_CONFIG["early_stopping_patience"]})')
    
    # Output options
    output_group = finetune_parser.add_argument_group('Output options')
    output_group.add_argument('--output-dir', type=str, default='./finetuned_models',
                            help='Output directory for fine-tuned models (default: ./finetuned_models)')
    output_group.add_argument('--run-name', type=str, default=None,
                            help='Name for this training run (default: auto-generated)')
    output_group.add_argument('--save-splits', action='store_true',
                            help='Save train/val/test splits to disk')
    
    # Hardware options
    hw_group = finetune_parser.add_argument_group('Hardware options')
    hw_group.add_argument('--no-fp16', action='store_true',
                        help='Disable mixed precision training')
    hw_group.add_argument('--cpu-only', action='store_true',
                        help='Force CPU-only training (disable CUDA)')
    
    # Miscellaneous
    misc_group = finetune_parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--random-state', type=int, default=42,
                          help='Random seed for reproducibility (default: 42)')
    misc_group.add_argument('--verbose', action='store_true',
                          help='Enable verbose logging')
    misc_group.add_argument('--dry-run', action='store_true',
                          help='Show configuration without training')
    misc_group.add_argument('--no-epoch-metrics', action='store_true',
                          help='Disable detailed metrics display at each epoch')
    
    return parser


def handle_interactive_command(args):
    """Handle the interactive fine-tuning command."""
    if args.demo:
        import subprocess
        subprocess.run([sys.executable, "demo_interactive.py"])
    else:
        import subprocess
        subprocess.run([sys.executable, "interactive_finetuning.py"])


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
    print(f"🚀 Starting SmolLM2 fine-tuning")
    print(f"📂 Data: {args.data_path}")
    
    # Determine model name
    if args.variant:
        variant_config = get_config_for_variant(args.variant)
        model_name = variant_config['model_name']
        print(f"🤖 Using variant: {args.variant} ({model_name})")
    else:
        model_name = args.model
        print(f"🤖 Using model: {model_name}")
    
    # Apply preset if specified
    training_config = {}
    if args.preset:
        preset_config = get_preset_config(args.preset)
        training_config.update(preset_config)
        print(f"⚙️ Using preset: {args.preset}")
    
    # Override with command line arguments
    cli_config = {
        'num_train_epochs': args.epochs,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
    }
    
    # Apply variant-specific overrides
    if args.variant:
        variant_config = get_config_for_variant(args.variant)
        if 'batch_size' in variant_config:
            cli_config['batch_size'] = variant_config['batch_size']
        if 'learning_rate' in variant_config:
            cli_config['learning_rate'] = variant_config['learning_rate']
    
    training_config.update(cli_config)
    
    # Hardware configuration
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("💻 Using CPU-only training")
    
    if args.no_fp16:
        training_config['fp16'] = False
        print("🔢 Disabled mixed precision training")
    
    # Show configuration
    print(f"\n📋 Training Configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    
    if args.dry_run:
        print("\n🔍 Dry run mode - configuration shown above")
        return
    
    # Initialize fine-tuner
    finetuner = SmolLM2FineTuner(
        model_name=model_name,
        output_dir=args.output_dir
    )
    
    try:
        # Run fine-tuning pipeline
        model_path, eval_results = finetuner.finetune_pipeline(
            data_path=args.data_path,
            test_size=args.test_size,
            val_size=args.val_size,
            text_column=args.text_column,
            label_column=args.label_column,
            run_name=args.run_name,
            show_epoch_metrics=not args.no_epoch_metrics,
            **training_config
        )
        
        print(f"\n🎉 Fine-tuning completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"📊 Final evaluation results:")
        for metric, value in eval_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        # Save results summary
        results_file = Path(model_path) / "results_summary.json"
        with open(results_file, 'w') as f:
            json.dump({
                'model_path': model_path,
                'training_config': training_config,
                'evaluation_results': eval_results,
                'data_path': args.data_path
            }, f, indent=2)
        
        print(f"📄 Results summary saved to: {results_file}")
        
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
    
    # Validate input file exists (not needed for interactive mode)
    if args.command != 'interactive' and hasattr(args, 'data_path'):
        if not os.path.exists(args.data_path):
            print(f"❌ Error: Data file not found: {args.data_path}")
            sys.exit(1)
    
    # Set logging level
    if hasattr(args, 'verbose') and args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Route to appropriate handler
    if args.command == 'interactive':
        handle_interactive_command(args)
    elif args.command == 'split':
        handle_split_command(args)
    elif args.command == 'finetune':
        handle_finetune_command(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()