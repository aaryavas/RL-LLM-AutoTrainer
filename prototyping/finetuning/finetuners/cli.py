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

# Add the parent directories to path for proper imports when running directly
_current_dir = Path(__file__).parent.resolve()
_finetuning_dir = _current_dir.parent  # finetuning/
_prototyping_dir = _finetuning_dir.parent  # prototyping/

# Insert prototyping dir so 'finetuning.finetuners' package works
if str(_prototyping_dir) not in sys.path:
    sys.path.insert(0, str(_prototyping_dir))

from finetuning.finetuners import SmolLM2VBLoRAFineTuner, split_synthetic_data
from finetuning.finetuners.config import (
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
    train_group.add_argument('--epochs', type=validate_positive_int, default=5,
                           help='Number of training epochs (default: 5)')
    train_group.add_argument('--lr', '--learning-rate', type=validate_positive_float, default=5e-4,
                           help='Base learning rate (default: 5e-4)')
    train_group.add_argument('--batch-size', type=validate_positive_int, default=4,
                           help='Batch size per device (default: 4)')
    train_group.add_argument('--early-stopping', type=validate_positive_int, default=2,
                           help='Early stopping patience (default: 2)')

    # VB-LoRA specific parameters
    vblora_group = finetune_parser.add_argument_group('VB-LoRA options')
    vblora_group.add_argument('--num-vectors', type=validate_positive_int, default=90,
                             help='Number of vectors in vector bank (default: 90)')
    vblora_group.add_argument('--vector-length', type=validate_positive_int, default=64,
                         help='Length of each vector - must divide model dimensions evenly (default: 64)')
    vblora_group.add_argument('--lora-r', type=validate_positive_int, default=16,
                             help='LoRA rank (default: 16)')
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
    
    # ORPO options
    orpo_group = finetune_parser.add_argument_group('ORPO improvement options')
    orpo_group.add_argument('--orpo', action='store_true',
                           help='Automatically run ORPO training after VB-LoRA')
    orpo_group.add_argument('--orpo-epochs', type=validate_positive_int, default=3,
                           help='Number of ORPO training epochs (default: 3)')
    orpo_group.add_argument('--orpo-beta', type=validate_positive_float, default=0.1,
                           help='ORPO beta parameter (default: 0.1)')
    orpo_group.add_argument('--orpo-lr', type=validate_positive_float, default=1e-5,
                           help='ORPO learning rate (default: 1e-5)')
    orpo_group.add_argument('--no-orpo-prompt', action='store_true',
                           help='Skip the ORPO improvement prompt after training')

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
        
        # Return model info for potential ORPO follow-up
        return model_path, model_name, args.data_path, config

    except Exception as e:
        print(f"❌ Fine-tuning failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def prompt_for_orpo_improvement() -> bool:
    """Ask user if they want to improve model with ORPO training."""
    print("\n" + "=" * 60)
    print("🔄 ORPO IMPROVEMENT OPTION")
    print("=" * 60)
    print("\nORPO (Odds Ratio Preference Optimization) can further improve")
    print("your model by training it to prefer correct responses over")
    print("its own mistakes.")
    print()
    
    while True:
        response = input("Would you like to run ORPO improvement training? [y/N]: ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no', '']:
            return False
        else:
            print("Please enter 'y' or 'n'")


def configure_orpo_training(default_epochs: int = 3, default_beta: float = 0.1) -> dict:
    """Interactive configuration for ORPO training parameters."""
    print("\n📋 ORPO Configuration")
    print("-" * 40)
    
    config = {}
    
    # Epochs
    try:
        epochs_input = input(f"  Number of ORPO epochs [{default_epochs}]: ").strip()
        config['epochs'] = int(epochs_input) if epochs_input else default_epochs
    except ValueError:
        config['epochs'] = default_epochs
    
    # Beta
    try:
        beta_input = input(f"  ORPO beta (preference strength) [{default_beta}]: ").strip()
        config['beta'] = float(beta_input) if beta_input else default_beta
    except ValueError:
        config['beta'] = default_beta
    
    # Learning rate
    try:
        lr_input = input("  Learning rate [1e-5]: ").strip()
        config['learning_rate'] = float(lr_input) if lr_input else 1e-5
    except ValueError:
        config['learning_rate'] = 1e-5
    
    # Batch size
    try:
        batch_input = input("  Batch size [2]: ").strip()
        config['batch_size'] = int(batch_input) if batch_input else 2
    except ValueError:
        config['batch_size'] = 2
    
    return config


def run_orpo_improvement(
    vblora_model_path: str,
    base_model_name: str,
    original_data_path: str,
    vblora_config: dict,
    orpo_config: dict,
    output_dir: str = "./output/orpo_models",
    verbose: bool = False,
):
    """
    Run the complete ORPO improvement pipeline.
    
    1. Merge VB-LoRA adapter into base model
    2. Generate rejected responses using the merged model
    3. Create ORPO preference dataset
    4. Train with ORPO
    """
    from finetuning.finetuners.utils.merge_adapter import merge_for_orpo
    from finetuning.finetuners.utils.orpo_generator import ORPODataGenerator
    from finetuning.finetuners.orpo.train_orpo import ORPOFineTuner
    import pandas as pd
    from datetime import datetime
    
    print("\n" + "=" * 60)
    print("🚀 STARTING ORPO IMPROVEMENT PIPELINE")
    print("=" * 60)
    
    # Step 1: Merge VB-LoRA adapter
    print("\n📦 Step 1: Merging VB-LoRA adapter weights...")
    try:
        merged_model_path, base_model = merge_for_orpo(
            vblora_model_path=vblora_model_path,
            base_model_name=base_model_name,
        )
        print(f"   ✅ Merged model saved to: {merged_model_path}")
    except Exception as e:
        print(f"   ❌ Failed to merge adapter: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return
    
    # Step 2: Generate rejected responses
    print("\n🔄 Step 2: Generating rejected responses from model...")
    try:
        # Load original data
        df = pd.read_csv(original_data_path)
        text_column = vblora_config.get('text_column', 'text')
        label_column = vblora_config.get('label_column', 'label')
        
        # Initialize ORPO data generator with the merged model
        generator = ORPODataGenerator(
            model_path=merged_model_path,
            load_in_4bit=True,
        )
        
        # Create ORPO dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        orpo_data_path = os.path.join(output_dir, f"orpo_preference_data_{timestamp}.csv")
        os.makedirs(output_dir, exist_ok=True)
        
        orpo_df = generator.create_orpo_dataset_from_synthetic(
            df=df,
            text_column=text_column,
            label_column=label_column,
            generate_rejected=True,
            output_path=orpo_data_path,
        )
        print(f"   ✅ Created ORPO dataset with {len(orpo_df)} examples")
        print(f"   📁 Saved to: {orpo_data_path}")
        
        # Clean up generator to free memory
        del generator
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"   ❌ Failed to generate ORPO data: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return
    
    # Step 3: Run ORPO training
    print("\n🎯 Step 3: Running ORPO training...")
    try:
        orpo_output_dir = os.path.join(output_dir, f"orpo_model_{timestamp}")
        
        tuner = ORPOFineTuner(
            model_name=merged_model_path,  # Use the merged model
            output_dir=orpo_output_dir,
        )
        
        result_path = tuner.finetune_pipeline(
            data_path=orpo_data_path,
            num_train_epochs=orpo_config.get('epochs', 3),
            learning_rate=orpo_config.get('learning_rate', 1e-5),
            batch_size=orpo_config.get('batch_size', 2),
            beta=orpo_config.get('beta', 0.1),
        )
        
        print(f"\n🎉 ORPO training completed!")
        print(f"📁 Final model saved to: {result_path}")
        
    except Exception as e:
        print(f"   ❌ ORPO training failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✅ ORPO IMPROVEMENT PIPELINE COMPLETED")
    print("=" * 60)


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
        result = handle_finetune_command(args)
        
        # Check if finetuning was successful and offer ORPO improvement
        if result is not None:
            model_path, model_name, data_path, config = result
            
            # Skip ORPO if dry-run
            if args.dry_run:
                return
            
            # Determine if ORPO should run
            run_orpo = False
            orpo_config = {}
            
            if args.orpo:
                # User explicitly requested ORPO
                run_orpo = True
                orpo_config = {
                    'epochs': args.orpo_epochs,
                    'beta': args.orpo_beta,
                    'learning_rate': args.orpo_lr,
                    'batch_size': 2,
                }
            elif not args.no_orpo_prompt:
                # Ask user interactively
                if prompt_for_orpo_improvement():
                    run_orpo = True
                    orpo_config = configure_orpo_training()
            
            if run_orpo:
                run_orpo_improvement(
                    vblora_model_path=model_path,
                    base_model_name=model_name,
                    original_data_path=data_path,
                    vblora_config=config,
                    orpo_config=orpo_config,
                    output_dir=args.output_dir,
                    verbose=args.verbose,
                )
    else:
        print(f"❌ Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
