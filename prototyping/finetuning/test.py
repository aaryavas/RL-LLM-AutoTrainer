#!/usr/bin/env python3
"""
Test script for SmolLM2 fine-tuning module.
This script validates the installation and basic functionality.
"""

import sys
import os
import tempfile
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"  ✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"  ❌ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"  ✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"  ❌ Transformers: {e}")
        return False
    
    try:
        import peft
        print(f"  ✅ PEFT: {peft.__version__}")
    except ImportError as e:
        print(f"  ❌ PEFT: {e}")
        return False
    
    try:
        import datasets
        print(f"  ✅ Datasets: {datasets.__version__}")
    except ImportError as e:
        print(f"  ❌ Datasets: {e}")
        return False
    
    try:
        import pandas
        print(f"  ✅ Pandas: {pandas.__version__}")
    except ImportError as e:
        print(f"  ❌ Pandas: {e}")
        return False
    
    try:
        import sklearn
        print(f"  ✅ Scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"  ❌ Scikit-learn: {e}")
        return False
    
    return True


def test_module_imports():
    """Test if our modules can be imported."""
    print("\n🔍 Testing module imports...")
    
    try:
        from finetuning import SmolLM2FineTuner, split_synthetic_data
        print("  ✅ Fine-tuning module imported successfully")
    except ImportError as e:
        print(f"  ❌ Fine-tuning module: {e}")
        return False
    
    try:
        from config import MODEL_VARIANTS, PRESETS, TRAINING_CONFIG
        print("  ✅ Config module imported successfully")
    except ImportError as e:
        print(f"  ❌ Config module: {e}")
        return False
    
    return True


def create_test_data(output_path: str, num_samples: int = 100):
    """Create a small test dataset."""
    print(f"\n📝 Creating test dataset with {num_samples} samples...")
    
    import random
    
    # Sample texts and labels
    positive_texts = [
        "This is a great example of positive sentiment",
        "I love this product, it works perfectly",
        "Excellent service and friendly staff",
        "Amazing quality and fast delivery",
        "Highly recommended, very satisfied"
    ]
    
    negative_texts = [
        "This is terrible and disappointing",
        "Poor quality and bad customer service",
        "Not worth the money, very upset",
        "Broken item and slow response",
        "Would not recommend to anyone"
    ]
    
    neutral_texts = [
        "This is an average product",
        "Standard quality as expected",
        "Nothing special but functional",
        "Meets basic requirements",
        "Typical experience, neither good nor bad"
    ]
    
    # Generate balanced dataset
    data = []
    for i in range(num_samples):
        if i % 3 == 0:
            text = random.choice(positive_texts) + f" (sample {i})"
            label = "positive"
        elif i % 3 == 1:
            text = random.choice(negative_texts) + f" (sample {i})"
            label = "negative"
        else:
            text = random.choice(neutral_texts) + f" (sample {i})"
            label = "neutral"
        
        data.append({
            "text": text,
            "label": label,
            "model": "test_model",
            "reasoning": "Generated for testing"
        })
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"  ✅ Test data saved to: {output_path}")
    print(f"  📊 Label distribution: {df['label'].value_counts().to_dict()}")
    
    return output_path


def test_data_splitting():
    """Test data splitting functionality."""
    print("\n🔍 Testing data splitting...")
    
    try:
        from finetuning import split_synthetic_data
        
        # Create temporary test data
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data_path = os.path.join(temp_dir, "test_data.csv")
            create_test_data(test_data_path, 60)
            
            # Test splitting
            output_dir = os.path.join(temp_dir, "split_data")
            train_path, val_path, test_path = split_synthetic_data(
                data_path=test_data_path,
                output_dir=output_dir,
                test_size=0.2,
                val_size=0.1
            )
            
            # Verify files exist
            assert os.path.exists(train_path), f"Train file not found: {train_path}"
            assert os.path.exists(val_path), f"Val file not found: {val_path}"
            assert os.path.exists(test_path), f"Test file not found: {test_path}"
            
            # Check data sizes
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            
            total_samples = len(train_df) + len(val_df) + len(test_df)
            print(f"  📊 Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            print(f"  📊 Total: {total_samples} samples")
            
            assert total_samples == 60, f"Expected 60 total samples, got {total_samples}"
            
        print("  ✅ Data splitting test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Data splitting test failed: {e}")
        return False


def test_finetuner_initialization():
    """Test fine-tuner initialization."""
    print("\n🔍 Testing fine-tuner initialization...")
    
    try:
        from finetuning import SmolLM2FineTuner
        
        # Test initialization (without loading model)
        finetuner = SmolLM2FineTuner(
            model_name="HuggingFaceTB/SmolLM2-135M-Instruct",  # Smallest model
            output_dir="./test_output"
        )
        
        print("  ✅ Fine-tuner initialized successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Fine-tuner initialization failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\n🔍 Testing configuration loading...")
    
    try:
        from config import (
            MODEL_VARIANTS, PRESETS, get_config_for_variant, 
            get_preset_config, merge_configs
        )
        
        # Test variant config
        variant_config = get_config_for_variant("SmolLM2-135M")
        assert "model_name" in variant_config
        print(f"  ✅ Variant config loaded: {variant_config}")
        
        # Test preset config
        preset_config = get_preset_config("quick_test")
        assert "num_train_epochs" in preset_config
        print(f"  ✅ Preset config loaded: {preset_config}")
        
        # Test config merging
        merged = merge_configs(variant_config, preset_config)
        assert len(merged) > 0
        print(f"  ✅ Config merging works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")
        return False


def test_cli_help():
    """Test CLI help functionality."""
    print("\n🔍 Testing CLI help...")
    
    try:
        import subprocess
        import sys
        
        # Test basic help
        result = subprocess.run([
            sys.executable, "cli.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "SmolLM2" in result.stdout:
            print("  ✅ CLI help works")
            return True
        else:
            print(f"  ❌ CLI help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ CLI help test failed: {e}")
        return False


def check_cuda_availability():
    """Check CUDA availability."""
    print("\n🔍 Checking CUDA availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"    Device {i}: {device_name} ({memory:.1f} GB)")
        else:
            print("  ⚠️ CUDA not available, will use CPU training")
        
        return True
        
    except Exception as e:
        print(f"  ❌ CUDA check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 SmolLM2 Fine-tuning Module Test Suite")
    print("=" * 50)
    
    tests = [
        ("Package imports", test_imports),
        ("Module imports", test_module_imports),
        ("Data splitting", test_data_splitting),
        ("Fine-tuner initialization", test_finetuner_initialization),
        ("Configuration loading", test_config_loading),
        ("CLI help", test_cli_help),
        ("CUDA availability", check_cuda_availability),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The module is ready to use.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the installation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())