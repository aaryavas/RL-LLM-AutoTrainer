#!/usr/bin/env python3
"""
Complete pipeline example: Data generation -> Fine-tuning
This script demonstrates the full workflow from synthetic data generation to model fine-tuning.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout.strip())
    else:
        print(f"❌ {description} failed")
        print("Error:", result.stderr.strip())
        return False
    
    return True

def generate_synthetic_data(config_path, sample_size=1000, output_dir="../generated_data"):
    """Generate synthetic data using the data generator."""
    print("📊 Step 1: Generating Synthetic Data")
    print("=" * 40)
    
    cmd = [
        "python3", "../data-gen.py",
        "--config", config_path,
        "--sample_size", str(sample_size),
        "--output_dir", output_dir,
        "--batch_size", "50",
        "--save_reasoning"
    ]
    
    return run_command(cmd, "Synthetic data generation")

def split_data(data_path, output_dir="./split_data"):
    """Split the generated data into train/val/test."""
    print("\n📈 Step 2: Splitting Data")
    print("=" * 40)
    
    cmd = [
        "python3", "cli.py", "split", data_path,
        "--output-dir", output_dir,
        "--test-size", "0.2",
        "--val-size", "0.1"
    ]
    
    return run_command(cmd, "Data splitting")

def finetune_model(data_path, variant="SmolLM2-360M", preset="quick_test"):
    """Fine-tune SmolLM2 on the synthetic data."""
    print("\n🤖 Step 3: Fine-tuning SmolLM2")
    print("=" * 40)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"synthetic_finetune_{timestamp}"
    
    cmd = [
        "python3", "cli.py", "finetune", data_path,
        "--variant", variant,
        "--preset", preset,
        "--run-name", run_name,
        "--save-splits"
    ]
    
    return run_command(cmd, f"Model fine-tuning ({variant}, {preset})")

def full_pipeline_example():
    """Run the complete pipeline."""
    print("🚀 Complete SmolLM2 Fine-tuning Pipeline")
    print("=" * 50)
    
    # Configuration
    config_path = "../intel-synthetic/polite-guard/config/polite-guard-config.py"
    sample_size = 500  # Small for quick testing
    
    # Check if config exists (using a fallback if not)
    if not os.path.exists(config_path):
        print(f"⚠️ Config file not found: {config_path}")
        print("Using default data generation settings...")
        config_path = "./config/polite-guard-config.py"  # Would need to create this
    
    # Step 1: Generate synthetic data
    output_dir = "../generated_data"
    if not generate_synthetic_data(config_path, sample_size, output_dir):
        return False
    
    # Find the most recent generated file
    data_files = list(Path(output_dir).glob("*.csv"))
    if not data_files:
        print("❌ No data files found in output directory")
        return False
    
    latest_data = max(data_files, key=os.path.getctime)
    print(f"📂 Using data file: {latest_data}")
    
    # Step 2: Split data (optional, fine-tuning will do this automatically)
    # split_data(str(latest_data))
    
    # Step 3: Fine-tune model
    if not finetune_model(str(latest_data), variant="SmolLM2-360M", preset="quick_test"):
        return False
    
    print("\n🎉 Complete pipeline finished successfully!")
    print("\nNext steps:")
    print("1. Check the finetuned_models/ directory for your trained model")
    print("2. Review the training metrics and evaluation results")
    print("3. Load the model for inference using PEFT")
    
    return True

def quick_test_example():
    """Quick test with dummy data."""
    print("🧪 Quick Test Example")
    print("=" * 30)
    
    # Create dummy data for testing
    import pandas as pd
    
    dummy_data = pd.DataFrame({
        'text': [
            "This is a positive example",
            "This is a negative example", 
            "Another positive case",
            "Another negative case",
            "Neutral statement here"
        ] * 20,  # Repeat to get 100 samples
        'label': ['positive', 'negative', 'positive', 'negative', 'neutral'] * 20,
        'model': 'dummy_model',
        'reasoning': 'Generated for testing'
    })
    
    # Save dummy data
    dummy_path = "./dummy_test_data.csv"
    dummy_data.to_csv(dummy_path, index=False)
    print(f"📝 Created dummy data: {dummy_path}")
    
    # Test fine-tuning with minimal settings
    return finetune_model(dummy_path, variant="SmolLM2-135M", preset="quick_test")

def show_usage():
    """Show usage examples."""
    print("📖 SmolLM2 Fine-tuning Pipeline Examples")
    print("=" * 50)
    print()
    print("1. Quick test with dummy data:")
    print("   python3 pipeline_example.py --quick-test")
    print()
    print("2. Full pipeline with synthetic data generation:")
    print("   python3 pipeline_example.py --full-pipeline")
    print()
    print("3. Fine-tune existing data:")
    print("   python3 cli.py finetune your_data.csv --variant SmolLM2-360M")
    print()
    print("4. Split data only:")
    print("   python3 cli.py split your_data.csv --output-dir ./splits")
    print()
    print("Available model variants:")
    print("  - SmolLM2-135M (fastest, least memory)")
    print("  - SmolLM2-360M (balanced)")
    print("  - SmolLM2-1.7B (best quality, most memory)")
    print()
    print("Available presets:")
    print("  - quick_test (1 epoch, for testing)")
    print("  - standard (3 epochs, balanced)")
    print("  - thorough (5 epochs, best quality)")
    print("  - memory_efficient (optimized for low memory)")

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick-test":
            quick_test_example()
        elif sys.argv[1] == "--full-pipeline":
            full_pipeline_example()
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            show_usage()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            show_usage()
    else:
        show_usage()

if __name__ == "__main__":
    main()