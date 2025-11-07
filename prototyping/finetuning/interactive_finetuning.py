#!/usr/bin/env python3
"""
Interactive SmolLM2 Fine-tuning Tool
This script guides users through the fine-tuning process step by step with questions.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from finetuning import SmolLM2FineTuner, split_synthetic_data
from config import (
    MODEL_VARIANTS, PRESETS, get_config_for_variant, 
    get_preset_config, merge_configs, TRAINING_CONFIG
)


class InteractiveFineTuner:
    """Interactive fine-tuning tool that guides users through the process."""
    
    def __init__(self):
        self.data_path = None
        self.model_variant = None
        self.preset = None
        self.custom_config = {}
        self.output_dir = "./finetuned_models"
        self.run_name = None
        self.advanced_mode = False
        self.show_epoch_metrics = True  # Default to showing metrics
        
    def print_header(self):
        """Print the welcome header."""
        print("\n" + "="*60)
        print("🤖 SmolLM2 Interactive Fine-tuning Tool")
        print("="*60)
        print("This tool will guide you through fine-tuning SmolLM2 models")
        print("using LoRA PEFT on your synthetic data step by step.")
        print("="*60)
    
    def validate_file_path(self, path: str) -> bool:
        """Validate that a file path exists."""
        return os.path.exists(path) and os.path.isfile(path)
    
    def validate_csv_data(self, path: str) -> Tuple[bool, Optional[str]]:
        """Validate CSV data format."""
        try:
            df = pd.read_csv(path)
            
            # Check if required columns exist
            required_cols = ['text', 'label']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
            
            # Check for empty data
            if len(df) == 0:
                return False, "CSV file is empty"
            
            # Check for missing values in required columns
            text_missing = df['text'].isna().sum()
            label_missing = df['label'].isna().sum()
            
            if text_missing > 0 or label_missing > 0:
                return False, f"Missing values found - text: {text_missing}, label: {label_missing}"
            
            # Check label distribution
            label_counts = df['label'].value_counts()
            if len(label_counts) < 2:
                return False, "Need at least 2 different labels for classification"
            
            return True, None
            
        except Exception as e:
            return False, f"Error reading CSV: {str(e)}"
    
    def get_data_info(self, path: str) -> Dict:
        """Get information about the dataset."""
        df = pd.read_csv(path)
        return {
            'total_samples': len(df),
            'labels': df['label'].value_counts().to_dict(),
            'num_labels': df['label'].nunique(),
            'columns': list(df.columns),
            'sample_texts': df['text'].head(3).tolist()
        }
    
    def step_1_data_selection(self):
        """Step 1: Select and validate data file."""
        print("\n🗂️  STEP 1: DATA SELECTION")
        print("-" * 30)
        print("First, let's select your synthetic data file for fine-tuning.")
        print()
        
        while True:
            # Check for recent data files
            recent_files = []
            data_dirs = ["../generated_data", "./", "../"]
            
            for data_dir in data_dirs:
                if os.path.exists(data_dir):
                    csv_files = list(Path(data_dir).glob("*.csv"))
                    csv_files = [f for f in csv_files if f.stat().st_size > 0]  # Non-empty files
                    csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Most recent first
                    recent_files.extend(csv_files[:3])  # Top 3 from each directory
            
            # Remove duplicates and limit to 5 most recent
            seen = set()
            unique_recent = []
            for f in recent_files:
                if f.name not in seen:
                    unique_recent.append(f)
                    seen.add(f.name)
                if len(unique_recent) >= 5:
                    break
            
            if unique_recent:
                print("Recent CSV files found:")
                for i, file_path in enumerate(unique_recent, 1):
                    size_kb = file_path.stat().st_size / 1024
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    print(f"  {i}. {file_path} ({size_kb:.1f} KB, {mtime.strftime('%Y-%m-%d %H:%M')})")
                print(f"  {len(unique_recent) + 1}. Enter custom path")
                print()
                
                choice = input(f"Select a file (1-{len(unique_recent) + 1}): ").strip()
                
                if choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(unique_recent):
                        self.data_path = str(unique_recent[choice_num - 1])
                    elif choice_num == len(unique_recent) + 1:
                        self.data_path = input("Enter the path to your CSV file: ").strip()
                    else:
                        print("❌ Invalid choice. Please try again.")
                        continue
                else:
                    self.data_path = choice
            else:
                print("No recent CSV files found.")
                self.data_path = input("Enter the path to your CSV file: ").strip()
            
            # Validate file path
            if not self.validate_file_path(self.data_path):
                print(f"❌ File not found: {self.data_path}")
                print("Please check the path and try again.")
                continue
            
            # Validate CSV format
            is_valid, error_msg = self.validate_csv_data(self.data_path)
            if not is_valid:
                print(f"❌ Invalid CSV format: {error_msg}")
                print("Please fix the data file and try again.")
                continue
            
            # Show data info
            data_info = self.get_data_info(self.data_path)
            print(f"\n✅ Data file validated successfully!")
            print(f"📊 Dataset Info:")
            print(f"  • Total samples: {data_info['total_samples']}")
            print(f"  • Number of labels: {data_info['num_labels']}")
            print(f"  • Label distribution: {data_info['labels']}")
            print(f"  • Columns: {', '.join(data_info['columns'])}")
            print(f"\n📝 Sample texts:")
            for i, text in enumerate(data_info['sample_texts'], 1):
                preview = text[:80] + "..." if len(text) > 80 else text
                print(f"  {i}. {preview}")
            
            confirm = input(f"\n✅ Use this data file? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                break
            else:
                print("Let's select a different file...")
    
    def step_2_model_selection(self):
        """Step 2: Select model variant."""
        print("\n🤖 STEP 2: MODEL SELECTION")
        print("-" * 30)
        print("Choose the SmolLM2 model variant to fine-tune.")
        print("Different variants have different memory requirements and capabilities.")
        print()
        
        variants = list(MODEL_VARIANTS.keys())
        memory_info = {
            "SmolLM2-135M": "~2 GB GPU memory, fastest training",
            "SmolLM2-360M": "~4 GB GPU memory, balanced performance",
            "SmolLM2-1.7B": "~8 GB GPU memory, best quality"
        }
        
        print("Available model variants:")
        for i, variant in enumerate(variants, 1):
            config = MODEL_VARIANTS[variant]
            memory = memory_info.get(variant, "Memory info not available")
            print(f"  {i}. {variant}")
            print(f"     Model: {config['model_name']}")
            print(f"     Memory: {memory}")
            print(f"     Recommended batch size: {config.get('batch_size', 'N/A')}")
            print()
        
        while True:
            choice = input(f"Select model variant (1-{len(variants)}): ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(variants):
                self.model_variant = variants[int(choice) - 1]
                variant_config = get_config_for_variant(self.model_variant)
                
                print(f"\n✅ Selected: {self.model_variant}")
                print(f"📋 Configuration:")
                for key, value in variant_config.items():
                    print(f"  • {key}: {value}")
                break
            else:
                print("❌ Invalid choice. Please select a number from the list.")
    
    def step_3_training_preset(self):
        """Step 3: Select training preset or custom configuration."""
        print("\n⚙️  STEP 3: TRAINING CONFIGURATION")
        print("-" * 35)
        print("Choose a training preset or configure custom settings.")
        print()
        
        presets = list(PRESETS.keys())
        preset_descriptions = {
            "quick_test": "1 epoch, minimal training for testing (fastest)",
            "standard": "3 epochs, balanced training for general use",
            "thorough": "5 epochs, comprehensive training for best results",
            "memory_efficient": "Optimized settings for limited GPU memory"
        }
        
        print("Available presets:")
        for i, preset in enumerate(presets, 1):
            config = PRESETS[preset]
            description = preset_descriptions.get(preset, "No description")
            print(f"  {i}. {preset.upper()}")
            print(f"     {description}")
            print(f"     Epochs: {config.get('num_train_epochs', 'N/A')}")
            print(f"     Batch size: {config.get('per_device_train_batch_size', 'N/A')}")
            print()
        
        print(f"  {len(presets) + 1}. CUSTOM - Configure settings manually")
        print()
        
        while True:
            choice = input(f"Select configuration (1-{len(presets) + 1}): ").strip()
            
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(presets):
                    self.preset = presets[choice_num - 1]
                    preset_config = get_preset_config(self.preset)
                    
                    print(f"\n✅ Selected preset: {self.preset.upper()}")
                    print(f"📋 Configuration:")
                    for key, value in preset_config.items():
                        print(f"  • {key}: {value}")
                    break
                    
                elif choice_num == len(presets) + 1:
                    self.advanced_mode = True
                    self.configure_custom_settings()
                    break
                else:
                    print("❌ Invalid choice. Please select a number from the list.")
            else:
                print("❌ Invalid input. Please enter a number.")
    
    def configure_custom_settings(self):
        """Configure custom training settings."""
        print("\n🔧 CUSTOM CONFIGURATION")
        print("-" * 25)
        print("Configure training parameters manually.")
        print("Press Enter to use default values shown in brackets.")
        print()
        
        # Get default values
        defaults = TRAINING_CONFIG.copy()
        if self.model_variant:
            variant_config = get_config_for_variant(self.model_variant)
            if 'batch_size' in variant_config:
                defaults['per_device_train_batch_size'] = variant_config['batch_size']
            if 'learning_rate' in variant_config:
                defaults['learning_rate'] = variant_config['learning_rate']
        
        # Configure each setting
        settings = [
            ('num_train_epochs', 'Number of training epochs', int, defaults['num_train_epochs']),
            ('learning_rate', 'Learning rate', float, defaults['learning_rate']),
            ('per_device_train_batch_size', 'Batch size per device', int, defaults['per_device_train_batch_size']),
            ('warmup_steps', 'Warmup steps', int, defaults['warmup_steps']),
            ('weight_decay', 'Weight decay', float, defaults['weight_decay']),
            ('early_stopping_patience', 'Early stopping patience', int, defaults['early_stopping_patience'])
        ]
        
        for setting_key, description, setting_type, default in settings:
            while True:
                try:
                    user_input = input(f"{description} [{default}]: ").strip()
                    if not user_input:
                        self.custom_config[setting_key] = default
                        break
                    else:
                        value = setting_type(user_input)
                        if setting_type == int and value <= 0:
                            print("❌ Value must be positive")
                            continue
                        if setting_type == float and value <= 0:
                            print("❌ Value must be positive")
                            continue
                        self.custom_config[setting_key] = value
                        break
                except ValueError:
                    print(f"❌ Invalid {setting_type.__name__}. Please try again.")
        
        print(f"\n✅ Custom configuration set:")
        for key, value in self.custom_config.items():
            print(f"  • {key}: {value}")
    
    def step_4_output_configuration(self):
        """Step 4: Configure output settings."""
        print("\n📁 STEP 4: OUTPUT CONFIGURATION")
        print("-" * 32)
        print("Configure where to save the fine-tuned model.")
        print()
        
        # Output directory
        default_dir = "./finetuned_models"
        user_dir = input(f"Output directory [{default_dir}]: ").strip()
        self.output_dir = user_dir if user_dir else default_dir
        
        # Create directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"smollm2_finetune_{timestamp}"
        user_name = input(f"Run name [{default_name}]: ").strip()
        self.run_name = user_name if user_name else default_name
        
        # Epoch metrics display option
        show_metrics = input("Show detailed metrics at each epoch? (Y/n): ").strip().lower()
        if show_metrics in ['n', 'no']:
            self.show_epoch_metrics = False
            print("⚠️ Epoch metrics display disabled")
        else:
            self.show_epoch_metrics = True
            print("✅ Will show detailed metrics at each epoch")
        
        print(f"\n✅ Output configuration:")
        print(f"  • Directory: {self.output_dir}")
        print(f"  • Run name: {self.run_name}")
        print(f"  • Full path: {os.path.join(self.output_dir, self.run_name)}")
        print(f"  • Epoch metrics: {'Enabled' if self.show_epoch_metrics else 'Disabled'}")
    
    def step_5_hardware_options(self):
        """Step 5: Configure hardware options."""
        print("\n💻 STEP 5: HARDWARE OPTIONS")
        print("-" * 29)
        
        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"✅ CUDA available with {gpu_count} GPU(s):")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                    print(f"  • Device {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
                use_gpu = input("\nUse GPU for training? (Y/n): ").strip().lower()
                if use_gpu in ['n', 'no']:
                    self.custom_config['cpu_only'] = True
                    print("⚠️ Using CPU-only training (will be much slower)")
                else:
                    print("✅ Using GPU acceleration")
                    
                    # Mixed precision option
                    use_fp16 = input("Use mixed precision training (faster, less memory)? (Y/n): ").strip().lower()
                    if use_fp16 in ['n', 'no']:
                        self.custom_config['no_fp16'] = True
                        print("⚠️ Disabled mixed precision training")
                    else:
                        print("✅ Using mixed precision training")
            else:
                print("⚠️ CUDA not available - will use CPU training")
                self.custom_config['cpu_only'] = True
        
        except ImportError:
            print("⚠️ PyTorch not available - assuming CPU training")
            self.custom_config['cpu_only'] = True
    
    def step_6_review_and_confirm(self):
        """Step 6: Review configuration and confirm."""
        print("\n📋 STEP 6: REVIEW CONFIGURATION")
        print("-" * 33)
        print("Please review your configuration before starting training.")
        print()
        
        # Build final configuration
        final_config = {}
        
        # Add preset or custom config
        if self.preset:
            preset_config = get_preset_config(self.preset)
            final_config.update(preset_config)
        
        # Add variant-specific settings
        if self.model_variant:
            variant_config = get_config_for_variant(self.model_variant)
            if 'learning_rate' in variant_config:
                final_config['learning_rate'] = variant_config['learning_rate']
        
        # Add custom settings
        final_config.update(self.custom_config)
        
        # Display configuration
        print("📊 DATASET:")
        data_info = self.get_data_info(self.data_path)
        print(f"  • File: {self.data_path}")
        print(f"  • Samples: {data_info['total_samples']}")
        print(f"  • Labels: {list(data_info['labels'].keys())}")
        
        print(f"\n🤖 MODEL:")
        if self.model_variant:
            variant_config = get_config_for_variant(self.model_variant)
            print(f"  • Variant: {self.model_variant}")
            print(f"  • Model: {variant_config['model_name']}")
        
        print(f"\n⚙️ TRAINING:")
        training_keys = ['num_train_epochs', 'learning_rate', 'per_device_train_batch_size', 
                        'warmup_steps', 'weight_decay', 'early_stopping_patience']
        for key in training_keys:
            if key in final_config:
                print(f"  • {key}: {final_config[key]}")
        
        if self.preset:
            print(f"  • Preset: {self.preset}")
        
        print(f"📁 OUTPUT:")
        print(f"  • Directory: {self.output_dir}")
        print(f"  • Run name: {self.run_name}")
        print(f"  • Epoch metrics: {'Enabled' if self.show_epoch_metrics else 'Disabled'}")
        
        print(f"\n💻 HARDWARE:")
        if final_config.get('cpu_only'):
            print("  • Using CPU-only training")
        else:
            print("  • Using GPU acceleration")
        if final_config.get('no_fp16'):
            print("  • Mixed precision: Disabled")
        else:
            print("  • Mixed precision: Enabled")
        
        print()
        confirm = input("🚀 Start fine-tuning with this configuration? (y/n): ").strip().lower()
        return confirm in ['y', 'yes']
    
    def run_finetuning(self):
        """Execute the fine-tuning process."""
        print("\n🚀 STARTING FINE-TUNING")
        print("=" * 30)
        
        try:
            # Initialize fine-tuner
            model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # Default
            if self.model_variant:
                variant_config = get_config_for_variant(self.model_variant)
                model_name = variant_config['model_name']
            
            finetuner = SmolLM2FineTuner(
                model_name=model_name,
                output_dir=self.output_dir
            )
            
            # Prepare arguments
            pipeline_args = {
                'data_path': self.data_path,
                'run_name': self.run_name,
                'show_epoch_metrics': self.show_epoch_metrics
            }
            
            # Add preset or custom config
            if self.preset:
                preset_config = get_preset_config(self.preset)
                # Map preset parameters to pipeline parameters
                if 'num_train_epochs' in preset_config:
                    pipeline_args['num_train_epochs'] = preset_config['num_train_epochs']
            
            # Add custom config
            for key, value in self.custom_config.items():
                if key in ['num_train_epochs', 'learning_rate', 'batch_size']:
                    pipeline_args[key] = value
            
            # Add variant-specific settings
            if self.model_variant:
                variant_config = get_config_for_variant(self.model_variant)
                if 'learning_rate' in variant_config and 'learning_rate' not in pipeline_args:
                    pipeline_args['learning_rate'] = variant_config['learning_rate']
                if 'batch_size' in variant_config and 'batch_size' not in pipeline_args:
                    pipeline_args['batch_size'] = variant_config['batch_size']
            
            print(f"📋 Final parameters: {pipeline_args}")
            
            # Handle CPU-only mode
            if self.custom_config.get('cpu_only'):
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                print("💻 Running in CPU-only mode")
            
            # Run the fine-tuning pipeline
            model_path, eval_results = finetuner.finetune_pipeline(**pipeline_args)
            
            print(f"\n🎉 FINE-TUNING COMPLETED SUCCESSFULLY!")
            print(f"📁 Model saved to: {model_path}")
            print(f"📊 Final evaluation results:")
            for metric, value in eval_results.items():
                if isinstance(value, float):
                    print(f"  • {metric}: {value:.4f}")
                else:
                    print(f"  • {metric}: {value}")
            
            # Save configuration for reference
            config_file = os.path.join(model_path, "interactive_config.json")
            config_data = {
                'data_path': self.data_path,
                'model_variant': self.model_variant,
                'preset': self.preset,
                'custom_config': self.custom_config,
                'output_dir': self.output_dir,
                'run_name': self.run_name,
                'pipeline_args': pipeline_args,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"📄 Configuration saved to: {config_file}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ FINE-TUNING FAILED:")
            print(f"Error: {str(e)}")
            
            # Ask if user wants to see full traceback
            show_traceback = input("\nShow detailed error information? (y/n): ").strip().lower()
            if show_traceback in ['y', 'yes']:
                import traceback
                traceback.print_exc()
            
            return False
    
    def run(self):
        """Run the complete interactive fine-tuning process."""
        self.print_header()
        
        try:
            # Step-by-step process
            self.step_1_data_selection()
            self.step_2_model_selection() 
            self.step_3_training_preset()
            self.step_4_output_configuration()
            self.step_5_hardware_options()
            
            if self.step_6_review_and_confirm():
                success = self.run_finetuning()
                
                if success:
                    print(f"\n✨ All done! Your fine-tuned model is ready to use.")
                    print(f"🔗 You can load it using:")
                    print(f"   from peft import PeftModel")
                    print(f"   model = PeftModel.from_pretrained(base_model, '{os.path.join(self.output_dir, self.run_name)}')")
                else:
                    print(f"\n💔 Fine-tuning was not successful. Please check the errors above.")
            else:
                print(f"\n⏹️ Fine-tuning cancelled by user.")
        
        except KeyboardInterrupt:
            print(f"\n\n⏹️ Process interrupted by user. Goodbye!")
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point for interactive fine-tuning."""
    interactive_finetuner = InteractiveFineTuner()
    interactive_finetuner.run()


if __name__ == "__main__":
    main()