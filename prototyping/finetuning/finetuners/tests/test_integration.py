#!/usr/bin/env python3
"""
Integration tests for the complete pipeline.
Tests the flow from data generation through VB-LoRA to ORPO.
"""

import sys
import os
import subprocess
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Add parent directories to path
_current_dir = Path(__file__).parent.resolve()
_finetuners_dir = _current_dir.parent
_finetuning_dir = _finetuners_dir.parent
_prototyping_dir = _finetuning_dir.parent

sys.path.insert(0, str(_prototyping_dir))

import pandas as pd


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def pass_test(self, name: str):
        self.passed += 1
        print(f"   ✅ {name}")
    
    def fail_test(self, name: str, reason: str = ""):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"   ❌ {name}: {reason}")
    
    def summary(self):
        print(f"\n{'='*60}")
        print(f"📊 Results: {self.passed} passed, {self.failed} failed")
        if self.errors:
            print("\nFailed tests:")
            for name, reason in self.errors:
                print(f"  - {name}: {reason}")
        print("="*60)
        return self.failed == 0


class TestDataPipeline:
    """Tests for data processing pipeline."""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def test_synthetic_data_format(self):
        """Test synthetic data format is correct."""
        # Check existing synthetic data
        data_path = _prototyping_dir / "generated_data" / "20251125_212959.csv"
        
        if not data_path.exists():
            self.results.fail_test("Synthetic data exists", f"File not found: {data_path}")
            return
        
        df = pd.read_csv(data_path)
        
        required_cols = ['text', 'label']
        has_cols = all(col in df.columns for col in required_cols)
        
        if has_cols and len(df) > 0:
            self.results.pass_test("Synthetic data format correct")
        else:
            self.results.fail_test("Synthetic data format correct", 
                                  f"Columns: {list(df.columns)}")
    
    def test_data_splitter_proportions(self):
        """Test data splitter creates correct proportions."""
        from finetuning.finetuners.utils.data_splitter import DataSplitter
        
        # Create sample data
        df = pd.DataFrame({
            'text': [f'text {i}' for i in range(1000)],
            'label': [i % 5 for i in range(1000)]
        })
        
        splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)
        train, val, test = splitter.split_dataframe(df, label_column='label')
        
        # Check proportions (with some tolerance)
        test_prop = len(test) / 1000
        val_prop = len(val) / (len(train) + len(val))
        
        test_ok = 0.15 <= test_prop <= 0.25
        val_ok = 0.05 <= val_prop <= 0.15
        
        if test_ok and val_ok:
            self.results.pass_test("Data split proportions correct")
        else:
            self.results.fail_test("Data split proportions correct", 
                                  f"test: {test_prop:.2f}, val: {val_prop:.2f}")
    
    def test_data_splitter_stratification(self):
        """Test data splitter maintains label distribution."""
        from finetuning.finetuners.utils.data_splitter import DataSplitter
        
        # Create imbalanced data
        labels = [0] * 100 + [1] * 300 + [2] * 600
        df = pd.DataFrame({
            'text': [f'text {i}' for i in range(1000)],
            'label': labels
        })
        
        splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)
        train, val, test = splitter.split_dataframe(df, label_column='label')
        
        # Check label distribution is roughly maintained
        original_dist = df['label'].value_counts(normalize=True)
        test_dist = test['label'].value_counts(normalize=True)
        
        # Allow 10% tolerance
        diffs = [abs(original_dist.get(i, 0) - test_dist.get(i, 0)) for i in range(3)]
        
        if all(d < 0.1 for d in diffs):
            self.results.pass_test("Data stratification maintained")
        else:
            self.results.fail_test("Data stratification maintained", 
                                  f"Distribution diffs: {diffs}")
    
    def test_orpo_data_conversion(self):
        """Test ORPO data conversion from synthetic format."""
        from finetuning.finetuners.utils.orpo_generator import ORPODataGenerator
        
        # Create synthetic data format
        df = pd.DataFrame({
            'text': ['What is AI?', 'Explain ML', 'Define DL'],
            'label': ['AI answer', 'ML answer', 'DL answer'],
            'model': ['test'] * 3,
            'reasoning': ['reason'] * 3,
        })
        
        rejected_df = pd.DataFrame({
            'rejected_response': ['wrong AI', 'wrong ML', 'wrong DL']
        })
        
        generator = ORPODataGenerator()
        orpo_df = generator.create_orpo_dataset(
            df=df,
            rejected_df=rejected_df,
            prompt_col='text',
            chosen_col='label',
            rejected_col='rejected_response'
        )
        
        # Validate output
        checks = [
            len(orpo_df) == 3,
            'prompt' in orpo_df.columns,
            'chosen' in orpo_df.columns,
            'rejected' in orpo_df.columns,
        ]
        
        if all(checks):
            self.results.pass_test("ORPO data conversion")
        else:
            self.results.fail_test("ORPO data conversion", 
                                  f"Columns: {list(orpo_df.columns)}, Len: {len(orpo_df)}")
    
    def test_orpo_handles_missing_data(self):
        """Test ORPO generator handles missing data gracefully."""
        from finetuning.finetuners.utils.orpo_generator import ORPODataGenerator
        
        # Create data with some NaN values
        df = pd.DataFrame({
            'text': ['Q1', None, 'Q3', 'Q4'],
            'label': ['A1', 'A2', None, 'A4'],
        })
        
        rejected_df = pd.DataFrame({
            'rejected_response': ['R1', 'R2', 'R3', None]
        })
        
        generator = ORPODataGenerator()
        orpo_df = generator.create_orpo_dataset(
            df=df,
            rejected_df=rejected_df,
            prompt_col='text',
            chosen_col='label',
            rejected_col='rejected_response'
        )
        
        # Should only have 1 valid row (Q1/A1/R1)
        if len(orpo_df) == 1:
            self.results.pass_test("ORPO handles missing data")
        else:
            self.results.fail_test("ORPO handles missing data", 
                                  f"Expected 1 row, got {len(orpo_df)}")
    
    def run_all(self):
        """Run all data pipeline tests."""
        print("\n🧪 Data Pipeline Tests")
        print("-" * 40)
        
        self.test_synthetic_data_format()
        self.test_data_splitter_proportions()
        self.test_data_splitter_stratification()
        self.test_orpo_data_conversion()
        self.test_orpo_handles_missing_data()


class TestConfigIntegration:
    """Tests for configuration integration."""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def test_config_serialization(self):
        """Test configs can be serialized to dict."""
        from finetuning.finetuners.config.base_config import (
            DataConfig, TrainingConfig, HardwareConfig
        )
        
        try:
            data_config = DataConfig()
            training_config = TrainingConfig()
            hardware_config = HardwareConfig()
            
            # Convert to dict
            data_dict = vars(data_config)
            training_dict = vars(training_config)
            hardware_dict = vars(hardware_config)
            
            # Should all be dicts
            checks = [
                isinstance(data_dict, dict),
                isinstance(training_dict, dict),
                isinstance(hardware_dict, dict),
            ]
            
            if all(checks):
                self.results.pass_test("Config serialization")
            else:
                self.results.fail_test("Config serialization", "Not all are dicts")
        except Exception as e:
            self.results.fail_test("Config serialization", str(e))
    
    def test_config_modification(self):
        """Test configs can be modified."""
        from finetuning.finetuners.config.base_config import TrainingConfig
        
        config = TrainingConfig()
        original_epochs = config.num_train_epochs
        
        config.num_train_epochs = 10
        config.learning_rate = 1e-5
        
        checks = [
            config.num_train_epochs == 10,
            config.learning_rate == 1e-5,
            original_epochs != 10,
        ]
        
        if all(checks):
            self.results.pass_test("Config modification")
        else:
            self.results.fail_test("Config modification", "Values not updated")
    
    def test_preset_application(self):
        """Test preset configs can be applied."""
        from finetuning.finetuners.config.model_variants import get_preset_config
        
        preset = get_preset_config("quick_test")
        
        checks = [
            "num_train_epochs" in preset or "epochs" in preset,
            "learning_rate" in preset or "lr" in preset,
        ]
        
        if any(checks):
            self.results.pass_test("Preset application")
        else:
            self.results.fail_test("Preset application", f"Keys: {list(preset.keys())}")
    
    def test_variant_config(self):
        """Test variant configs provide model name."""
        from finetuning.finetuners.config.model_variants import get_variant_config
        
        variant = get_variant_config("SmolLM2-135M")
        
        if hasattr(variant, 'model_name') and "SmolLM2" in variant.model_name:
            self.results.pass_test("Variant config")
        else:
            self.results.fail_test("Variant config", "No model_name")
    
    def run_all(self):
        """Run all config integration tests."""
        print("\n🧪 Config Integration Tests")
        print("-" * 40)
        
        self.test_config_serialization()
        self.test_config_modification()
        self.test_preset_application()
        self.test_variant_config()


class TestMergeAdapter:
    """Tests for adapter merge functionality."""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def test_merge_adapter_functions_exist(self):
        """Test merge adapter functions exist."""
        from finetuning.finetuners.utils.merge_adapter import (
            merge_adapter, merge_for_orpo, get_base_model_from_adapter
        )
        
        checks = [
            callable(merge_adapter),
            callable(merge_for_orpo),
            callable(get_base_model_from_adapter),
        ]
        
        if all(checks):
            self.results.pass_test("Merge adapter functions exist")
        else:
            self.results.fail_test("Merge adapter functions exist", "Missing functions")
    
    def test_get_base_model_nonexistent(self):
        """Test get_base_model returns None for nonexistent path."""
        from finetuning.finetuners.utils.merge_adapter import get_base_model_from_adapter
        
        result = get_base_model_from_adapter("/nonexistent/path")
        
        if result is None:
            self.results.pass_test("get_base_model_from_adapter (nonexistent)")
        else:
            self.results.fail_test("get_base_model_from_adapter (nonexistent)", 
                                  f"Got: {result}")
    
    def test_get_base_model_from_config(self):
        """Test get_base_model reads from adapter_config.json."""
        from finetuning.finetuners.utils.merge_adapter import get_base_model_from_adapter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"base_model_name_or_path": "test/model"}
            config_path = os.path.join(temp_dir, "adapter_config.json")
            
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            result = get_base_model_from_adapter(temp_dir)
            
            if result == "test/model":
                self.results.pass_test("get_base_model_from_adapter (with config)")
            else:
                self.results.fail_test("get_base_model_from_adapter (with config)", 
                                      f"Got: {result}")
    
    def run_all(self):
        """Run all merge adapter tests."""
        print("\n🧪 Merge Adapter Tests")
        print("-" * 40)
        
        self.test_merge_adapter_functions_exist()
        self.test_get_base_model_nonexistent()
        self.test_get_base_model_from_config()


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def test_vblora_dry_run_pipeline(self):
        """Test VB-LoRA dry run shows full config."""
        # Create test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text,label\n")
            for i in range(10):
                f.write(f"sample text {i},label_{i % 3}\n")
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ["python", str(_finetuners_dir / "cli.py"), "finetune", temp_path,
                 "--dry-run", "--variant", "SmolLM2-135M"],
                capture_output=True,
                text=True,
                cwd=str(_prototyping_dir)
            )
            
            output = result.stdout + result.stderr
            
            # Should show configuration
            checks = [
                "Configuration" in output or "config" in output.lower(),
                "Dry run" in output,
            ]
            
            if all(checks) and result.returncode == 0:
                self.results.pass_test("VB-LoRA dry run pipeline")
            else:
                self.results.fail_test("VB-LoRA dry run pipeline", 
                                      f"Code: {result.returncode}, Output: {output[:200]}")
        finally:
            os.unlink(temp_path)
    
    def test_orpo_dry_run_pipeline(self):
        """Test ORPO dry run shows full config."""
        # Create ORPO test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("prompt,chosen,rejected\n")
            for i in range(5):
                f.write(f"question {i},correct {i},wrong {i}\n")
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ["python", str(_finetuners_dir / "orpo" / "cli.py"),
                 "--data_path", temp_path, "--dry-run"],
                capture_output=True,
                text=True,
                cwd=str(_prototyping_dir)
            )
            
            output = result.stdout + result.stderr
            
            checks = [
                "Configuration" in output or "config" in output.lower(),
                "Dry run" in output,
            ]
            
            if all(checks) and result.returncode == 0:
                self.results.pass_test("ORPO dry run pipeline")
            else:
                self.results.fail_test("ORPO dry run pipeline", 
                                      f"Code: {result.returncode}, Output: {output[:200]}")
        finally:
            os.unlink(temp_path)
    
    def test_split_then_finetune_dry_run(self):
        """Test split followed by finetune dry run."""
        # Create test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text,label\n")
            for i in range(100):
                f.write(f"sample text {i},label_{i % 3}\n")
            temp_path = f.name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # First split
                split_result = subprocess.run(
                    ["python", str(_finetuners_dir / "cli.py"), "split", temp_path,
                     "--output-dir", temp_dir],
                    capture_output=True,
                    text=True,
                    cwd=str(_prototyping_dir)
                )
                
                if split_result.returncode != 0:
                    self.results.fail_test("Split then finetune", 
                                          f"Split failed: {split_result.stderr}")
                    return
                
                # Then finetune dry run on train.csv
                train_path = os.path.join(temp_dir, "train.csv")
                finetune_result = subprocess.run(
                    ["python", str(_finetuners_dir / "cli.py"), "finetune", train_path,
                     "--dry-run", "--variant", "SmolLM2-135M"],
                    capture_output=True,
                    text=True,
                    cwd=str(_prototyping_dir)
                )
                
                if finetune_result.returncode == 0:
                    self.results.pass_test("Split then finetune pipeline")
                else:
                    self.results.fail_test("Split then finetune pipeline", 
                                          f"Finetune failed: {finetune_result.stderr[:200]}")
            finally:
                os.unlink(temp_path)
    
    def run_all(self):
        """Run all end-to-end tests."""
        print("\n🧪 End-to-End Tests")
        print("-" * 40)
        
        self.test_vblora_dry_run_pipeline()
        self.test_orpo_dry_run_pipeline()
        self.test_split_then_finetune_dry_run()


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("🧪 INTEGRATION TEST SUITE")
    print("=" * 60)
    
    results = TestResults()
    
    TestDataPipeline(results).run_all()
    TestConfigIntegration(results).run_all()
    TestMergeAdapter(results).run_all()
    TestEndToEnd(results).run_all()
    
    success = results.summary()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
