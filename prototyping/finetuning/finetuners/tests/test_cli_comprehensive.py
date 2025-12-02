#!/usr/bin/env python3
"""
Comprehensive test suite for the VB-LoRA CLI.
Tests all CLI arguments, configuration parsing, and validation.
"""

import sys
import os
import subprocess
import tempfile
import json
from pathlib import Path

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


def run_cli_command(args: list, cwd: str = None) -> tuple:
    """Run a CLI command and return (returncode, stdout, stderr)."""
    if cwd is None:
        cwd = str(_prototyping_dir)
    
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=cwd
    )
    return result.returncode, result.stdout, result.stderr


class TestVBLoRACLI:
    """Tests for VB-LoRA CLI."""
    
    def __init__(self, results: TestResults):
        self.results = results
        self.cli_path = str(_finetuners_dir / "cli.py")
    
    def test_help_command(self):
        """Test --help flag works."""
        code, stdout, stderr = run_cli_command(["python", self.cli_path, "--help"])
        if code == 0 and "Fine-tune SmolLM2 models" in stdout:
            self.results.pass_test("--help command")
        else:
            self.results.fail_test("--help command", f"Exit code: {code}")
    
    def test_split_help(self):
        """Test split subcommand help."""
        code, stdout, stderr = run_cli_command(["python", self.cli_path, "split", "--help"])
        if code == 0 and "output-dir" in stdout:
            self.results.pass_test("split --help")
        else:
            self.results.fail_test("split --help", f"Exit code: {code}")
    
    def test_finetune_help(self):
        """Test finetune subcommand help."""
        code, stdout, stderr = run_cli_command(["python", self.cli_path, "finetune", "--help"])
        if code == 0 and "VB-LoRA" in stdout:
            self.results.pass_test("finetune --help")
        else:
            self.results.fail_test("finetune --help", f"Exit code: {code}")
    
    def test_finetune_has_all_args(self):
        """Test finetune has all required arguments."""
        code, stdout, stderr = run_cli_command(["python", self.cli_path, "finetune", "--help"])
        
        required_args = [
            "--model", "--variant", "--preset",
            "--test-size", "--val-size", "--text-column", "--label-column",
            "--epochs", "--lr", "--batch-size", "--early-stopping",
            "--num-vectors", "--vector-length", "--lora-r",
            "--lr-vector-bank", "--lr-logits",
            "--output-dir", "--run-name",
            "--bits", "--bf16", "--fp16",
            "--verbose", "--dry-run", "--no-epoch-metrics",
            "--orpo", "--orpo-epochs", "--orpo-beta", "--orpo-lr"
        ]
        
        missing = []
        for arg in required_args:
            if arg not in stdout:
                missing.append(arg)
        
        if not missing:
            self.results.pass_test("All finetune arguments present")
        else:
            self.results.fail_test("All finetune arguments present", f"Missing: {missing}")
    
    def test_missing_data_file_error(self):
        """Test proper error for missing data file."""
        code, stdout, stderr = run_cli_command([
            "python", self.cli_path, "finetune", "/nonexistent/file.csv"
        ])
        
        if code != 0 and "not found" in (stdout + stderr).lower():
            self.results.pass_test("Missing file error handling")
        else:
            self.results.fail_test("Missing file error handling", "No proper error message")
    
    def test_dry_run_mode(self):
        """Test dry-run mode with valid data."""
        # Create temp CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text,label\n")
            f.write("test input,test output\n")
            temp_path = f.name
        
        try:
            code, stdout, stderr = run_cli_command([
                "python", self.cli_path, "finetune", temp_path,
                "--dry-run", "--variant", "SmolLM2-135M"
            ])
            
            if code == 0 and "Dry run mode" in stdout:
                self.results.pass_test("Dry-run mode")
            else:
                self.results.fail_test("Dry-run mode", f"Exit: {code}, Output: {stdout[:200]}")
        finally:
            os.unlink(temp_path)
    
    def test_preset_configurations(self):
        """Test preset configuration options."""
        code, stdout, stderr = run_cli_command(["python", self.cli_path, "finetune", "--help"])
        
        presets = ["quick_test", "standard", "thorough", "memory_efficient"]
        found = all(preset in stdout for preset in presets)
        
        if found:
            self.results.pass_test("Preset configurations available")
        else:
            self.results.fail_test("Preset configurations available", "Missing presets")
    
    def test_variant_options(self):
        """Test model variant options."""
        code, stdout, stderr = run_cli_command(["python", self.cli_path, "finetune", "--help"])
        
        variants = ["SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B"]
        found = all(variant in stdout for variant in variants)
        
        if found:
            self.results.pass_test("Model variants available")
        else:
            self.results.fail_test("Model variants available", "Missing variants")
    
    def test_split_command(self):
        """Test split command with valid data."""
        # Create temp CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text,label\n")
            for i in range(100):
                f.write(f"text {i},label {i % 5}\n")
            temp_path = f.name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                code, stdout, stderr = run_cli_command([
                    "python", self.cli_path, "split", temp_path,
                    "--output-dir", temp_dir
                ])
                
                # Check output files exist
                train_exists = os.path.exists(os.path.join(temp_dir, "train.csv"))
                val_exists = os.path.exists(os.path.join(temp_dir, "val.csv"))
                test_exists = os.path.exists(os.path.join(temp_dir, "test.csv"))
                
                if code == 0 and train_exists and val_exists and test_exists:
                    self.results.pass_test("Split command creates files")
                else:
                    self.results.fail_test("Split command creates files", 
                                          f"Exit: {code}, Files: {train_exists}/{val_exists}/{test_exists}")
            finally:
                os.unlink(temp_path)
    
    def run_all(self):
        """Run all VB-LoRA CLI tests."""
        print("\n🧪 VB-LoRA CLI Tests")
        print("-" * 40)
        
        self.test_help_command()
        self.test_split_help()
        self.test_finetune_help()
        self.test_finetune_has_all_args()
        self.test_missing_data_file_error()
        self.test_dry_run_mode()
        self.test_preset_configurations()
        self.test_variant_options()
        self.test_split_command()


class TestORPOCLI:
    """Tests for ORPO CLI."""
    
    def __init__(self, results: TestResults):
        self.results = results
        self.cli_path = str(_finetuners_dir / "orpo" / "cli.py")
    
    def test_help_command(self):
        """Test --help flag works."""
        code, stdout, stderr = run_cli_command(["python", self.cli_path, "--help"])
        if code == 0 and "ORPO" in stdout:
            self.results.pass_test("ORPO --help command")
        else:
            self.results.fail_test("ORPO --help command", f"Exit code: {code}")
    
    def test_all_arguments_present(self):
        """Test all ORPO arguments are present."""
        code, stdout, stderr = run_cli_command(["python", self.cli_path, "--help"])
        
        required_args = [
            "--model_name", "--data_path", "--output_dir",
            "--prompt_column", "--chosen_column", "--rejected_column",
            "--test_size", "--val_size",
            "--epochs", "--batch_size", "--lr", "--gradient_accumulation",
            "--beta", "--max_prompt_length", "--max_completion_length",
            "--bits", "--bf16", "--fp16",
            "--run_name", "--save_steps",
            "--verbose", "--dry-run", "--no-metrics"
        ]
        
        missing = [arg for arg in required_args if arg not in stdout]
        
        if not missing:
            self.results.pass_test("ORPO all arguments present")
        else:
            self.results.fail_test("ORPO all arguments present", f"Missing: {missing}")
    
    def test_dry_run_mode(self):
        """Test ORPO dry-run mode."""
        # Create temp preference CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("prompt,chosen,rejected\n")
            f.write("What is 2+2?,4,5\n")
            temp_path = f.name
        
        try:
            code, stdout, stderr = run_cli_command([
                "python", self.cli_path,
                "--data_path", temp_path,
                "--dry-run"
            ])
            
            if code == 0 and "Dry run mode" in stdout:
                self.results.pass_test("ORPO dry-run mode")
            else:
                self.results.fail_test("ORPO dry-run mode", f"Exit: {code}")
        finally:
            os.unlink(temp_path)
    
    def test_missing_file_error(self):
        """Test proper error for missing data file."""
        code, stdout, stderr = run_cli_command([
            "python", self.cli_path,
            "--data_path", "/nonexistent/file.csv"
        ])
        
        if code != 0 and "not found" in (stdout + stderr).lower():
            self.results.pass_test("ORPO missing file error")
        else:
            self.results.fail_test("ORPO missing file error", "No proper error")
    
    def run_all(self):
        """Run all ORPO CLI tests."""
        print("\n🧪 ORPO CLI Tests")
        print("-" * 40)
        
        self.test_help_command()
        self.test_all_arguments_present()
        self.test_dry_run_mode()
        self.test_missing_file_error()


class TestConfigModule:
    """Tests for configuration module."""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def test_base_config_imports(self):
        """Test base config can be imported."""
        try:
            from finetuning.finetuners.config.base_config import (
                DataConfig, TrainingConfig, OutputConfig, HardwareConfig
            )
            self.results.pass_test("Base config imports")
        except ImportError as e:
            self.results.fail_test("Base config imports", str(e))
    
    def test_vblora_config_imports(self):
        """Test VB-LoRA config can be imported."""
        try:
            from finetuning.finetuners.config.vblora_config import VBLoRAConfig, VBLoRADefaults
            self.results.pass_test("VB-LoRA config imports")
        except ImportError as e:
            self.results.fail_test("VB-LoRA config imports", str(e))
    
    def test_model_variants_imports(self):
        """Test model variants can be imported."""
        try:
            from finetuning.finetuners.config.model_variants import (
                SMOLLM2_VARIANTS, PRESET_CONFIGS, get_variant_config, get_preset_config
            )
            self.results.pass_test("Model variants imports")
        except ImportError as e:
            self.results.fail_test("Model variants imports", str(e))
    
    def test_data_config_defaults(self):
        """Test DataConfig has correct defaults."""
        from finetuning.finetuners.config.base_config import DataConfig
        
        config = DataConfig()
        checks = [
            config.test_size == 0.2,
            config.val_size == 0.1,
            config.text_column == "text",
            config.label_column == "label",
        ]
        
        if all(checks):
            self.results.pass_test("DataConfig defaults")
        else:
            self.results.fail_test("DataConfig defaults", f"Got: {config}")
    
    def test_training_config_defaults(self):
        """Test TrainingConfig has correct defaults."""
        from finetuning.finetuners.config.base_config import TrainingConfig
        
        config = TrainingConfig()
        checks = [
            config.num_train_epochs == 3,
            config.learning_rate == 5e-4,
            config.per_device_train_batch_size == 4,
            config.early_stopping_patience == 2,
        ]
        
        if all(checks):
            self.results.pass_test("TrainingConfig defaults")
        else:
            self.results.fail_test("TrainingConfig defaults", f"Got: {config}")
    
    def test_hardware_config_defaults(self):
        """Test HardwareConfig has correct defaults."""
        from finetuning.finetuners.config.base_config import HardwareConfig
        
        config = HardwareConfig()
        checks = [
            config.bits == 4,
            config.double_quant == True,
            config.quant_type == "nf4",
        ]
        
        if all(checks):
            self.results.pass_test("HardwareConfig defaults")
        else:
            self.results.fail_test("HardwareConfig defaults", f"Got: {config}")
    
    def test_config_validation(self):
        """Test config validation methods."""
        from finetuning.finetuners.config.base_config import DataConfig, HardwareConfig
        
        # Valid config should not raise
        try:
            config = DataConfig()
            config.validate()
            self.results.pass_test("Config validation (valid)")
        except Exception as e:
            self.results.fail_test("Config validation (valid)", str(e))
        
        # Invalid config should raise
        try:
            config = DataConfig(test_size=1.5)  # Invalid
            config.validate()
            self.results.fail_test("Config validation (invalid)", "Should have raised")
        except ValueError:
            self.results.pass_test("Config validation (invalid)")
    
    def test_variant_configs(self):
        """Test model variant configurations."""
        from finetuning.finetuners.config.model_variants import (
            SMOLLM2_VARIANTS, get_variant_config
        )
        
        expected_variants = ["SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B"]
        
        if all(v in SMOLLM2_VARIANTS for v in expected_variants):
            self.results.pass_test("Variant configurations exist")
        else:
            self.results.fail_test("Variant configurations exist", 
                                  f"Available: {list(SMOLLM2_VARIANTS.keys())}")
    
    def test_preset_configs(self):
        """Test preset configurations."""
        from finetuning.finetuners.config.model_variants import (
            PRESET_CONFIGS, get_preset_config
        )
        
        expected_presets = ["quick_test", "standard", "thorough", "memory_efficient"]
        
        if all(p in PRESET_CONFIGS for p in expected_presets):
            self.results.pass_test("Preset configurations exist")
        else:
            self.results.fail_test("Preset configurations exist", 
                                  f"Available: {list(PRESET_CONFIGS.keys())}")
    
    def run_all(self):
        """Run all config tests."""
        print("\n🧪 Configuration Module Tests")
        print("-" * 40)
        
        self.test_base_config_imports()
        self.test_vblora_config_imports()
        self.test_model_variants_imports()
        self.test_data_config_defaults()
        self.test_training_config_defaults()
        self.test_hardware_config_defaults()
        self.test_config_validation()
        self.test_variant_configs()
        self.test_preset_configs()


class TestUtilsModule:
    """Tests for utilities module."""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def test_data_splitter_import(self):
        """Test DataSplitter can be imported."""
        try:
            from finetuning.finetuners.utils.data_splitter import DataSplitter
            self.results.pass_test("DataSplitter import")
        except ImportError as e:
            self.results.fail_test("DataSplitter import", str(e))
    
    def test_data_splitter_split(self):
        """Test DataSplitter splits correctly."""
        from finetuning.finetuners.utils.data_splitter import DataSplitter
        
        # Create sample data
        df = pd.DataFrame({
            'text': [f'text {i}' for i in range(100)],
            'label': [i % 3 for i in range(100)]
        })
        
        splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)
        train, val, test = splitter.split_dataframe(df, label_column='label')
        
        # Check approximate sizes
        total = len(train) + len(val) + len(test)
        
        if total == 100 and len(test) > 0 and len(val) > 0 and len(train) > 0:
            self.results.pass_test("DataSplitter split")
        else:
            self.results.fail_test("DataSplitter split", 
                                  f"Sizes: train={len(train)}, val={len(val)}, test={len(test)}")
    
    def test_helpers_import(self):
        """Test helper functions can be imported."""
        try:
            from finetuning.finetuners.utils.helpers import ensure_dir, save_json, load_json
            self.results.pass_test("Helpers import")
        except ImportError as e:
            self.results.fail_test("Helpers import", str(e))
    
    def test_ensure_dir(self):
        """Test ensure_dir creates directories."""
        from finetuning.finetuners.utils.helpers import ensure_dir
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "nested", "dir", "path")
            ensure_dir(test_path)
            
            if os.path.isdir(test_path):
                self.results.pass_test("ensure_dir creates directories")
            else:
                self.results.fail_test("ensure_dir creates directories", "Directory not created")
    
    def test_save_load_json(self):
        """Test save_json and load_json."""
        from finetuning.finetuners.utils.helpers import save_json, load_json
        
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_json(test_data, temp_path)
            loaded = load_json(temp_path)
            
            if loaded == test_data:
                self.results.pass_test("save_json/load_json")
            else:
                self.results.fail_test("save_json/load_json", f"Data mismatch: {loaded}")
        finally:
            os.unlink(temp_path)
    
    def test_merge_adapter_import(self):
        """Test merge adapter can be imported."""
        try:
            from finetuning.finetuners.utils.merge_adapter import (
                merge_adapter, merge_for_orpo, get_base_model_from_adapter
            )
            self.results.pass_test("Merge adapter import")
        except ImportError as e:
            self.results.fail_test("Merge adapter import", str(e))
    
    def test_orpo_generator_import(self):
        """Test ORPO generator can be imported."""
        try:
            from finetuning.finetuners.utils.orpo_generator import ORPODataGenerator
            self.results.pass_test("ORPO generator import")
        except ImportError as e:
            self.results.fail_test("ORPO generator import", str(e))
    
    def run_all(self):
        """Run all utils tests."""
        print("\n🧪 Utilities Module Tests")
        print("-" * 40)
        
        self.test_data_splitter_import()
        self.test_data_splitter_split()
        self.test_helpers_import()
        self.test_ensure_dir()
        self.test_save_load_json()
        self.test_merge_adapter_import()
        self.test_orpo_generator_import()


class TestORPOModule:
    """Tests for ORPO module."""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def test_orpo_config_import(self):
        """Test ORPO config can be imported."""
        try:
            from finetuning.finetuners.orpo.config.orpo_config import ORPOSpecificConfig
            self.results.pass_test("ORPO config import")
        except ImportError as e:
            self.results.fail_test("ORPO config import", str(e))
    
    def test_orpo_config_defaults(self):
        """Test ORPO config has correct defaults."""
        from finetuning.finetuners.orpo.config.orpo_config import ORPOSpecificConfig
        
        config = ORPOSpecificConfig()
        checks = [
            config.beta == 0.5,
            config.max_prompt_length == 512,
            config.max_completion_length == 1024,
            config.disable_dropout == True,
        ]
        
        if all(checks):
            self.results.pass_test("ORPO config defaults")
        else:
            self.results.fail_test("ORPO config defaults", f"Got: {config}")
    
    def test_orpo_trainer_import(self):
        """Test ORPO trainer can be imported."""
        try:
            from finetuning.finetuners.orpo.train_orpo import ORPOFineTuner
            self.results.pass_test("ORPO trainer import")
        except ImportError as e:
            self.results.fail_test("ORPO trainer import", str(e))
    
    def test_orpo_data_processor_import(self):
        """Test ORPO data processor can be imported."""
        try:
            from finetuning.finetuners.orpo.core.data_processor import PreferenceDataProcessor
            self.results.pass_test("ORPO data processor import")
        except ImportError as e:
            self.results.fail_test("ORPO data processor import", str(e))
    
    def test_orpo_visualization_import(self):
        """Test ORPO visualization can be imported."""
        try:
            from finetuning.finetuners.orpo.training.visualization import TrainingVisualizer
            self.results.pass_test("ORPO visualization import")
        except ImportError as e:
            self.results.fail_test("ORPO visualization import", str(e))
    
    def run_all(self):
        """Run all ORPO module tests."""
        print("\n🧪 ORPO Module Tests")
        print("-" * 40)
        
        self.test_orpo_config_import()
        self.test_orpo_config_defaults()
        self.test_orpo_trainer_import()
        self.test_orpo_data_processor_import()
        self.test_orpo_visualization_import()


class TestCoreModule:
    """Tests for core module components."""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def test_tokenizer_manager_import(self):
        """Test TokenizerManager can be imported."""
        try:
            from finetuning.finetuners.core.tokenizer_manager import TokenizerManager
            self.results.pass_test("TokenizerManager import")
        except ImportError as e:
            self.results.fail_test("TokenizerManager import", str(e))
    
    def test_model_loader_import(self):
        """Test ModelLoader can be imported."""
        try:
            from finetuning.finetuners.core.model_loader import ModelLoader
            self.results.pass_test("ModelLoader import")
        except ImportError as e:
            self.results.fail_test("ModelLoader import", str(e))
    
    def test_data_processor_import(self):
        """Test DataProcessor can be imported."""
        try:
            from finetuning.finetuners.core.data_processor import DataProcessor
            self.results.pass_test("DataProcessor import")
        except ImportError as e:
            self.results.fail_test("DataProcessor import", str(e))
    
    def test_optimizer_factory_import(self):
        """Test OptimizerFactory can be imported."""
        try:
            from finetuning.finetuners.core.optimizer_factory import OptimizerFactory
            self.results.pass_test("OptimizerFactory import")
        except ImportError as e:
            self.results.fail_test("OptimizerFactory import", str(e))
    
    def run_all(self):
        """Run all core module tests."""
        print("\n🧪 Core Module Tests")
        print("-" * 40)
        
        self.test_tokenizer_manager_import()
        self.test_model_loader_import()
        self.test_data_processor_import()
        self.test_optimizer_factory_import()


class TestTrainingModule:
    """Tests for training module components."""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def test_trainer_import(self):
        """Test VBLoRATrainer can be imported."""
        try:
            from finetuning.finetuners.training.trainer import VBLoRATrainer
            self.results.pass_test("VBLoRATrainer import")
        except ImportError as e:
            self.results.fail_test("VBLoRATrainer import", str(e))
    
    def test_metrics_import(self):
        """Test MetricsComputer can be imported."""
        try:
            from finetuning.finetuners.training.metrics import MetricsComputer
            self.results.pass_test("MetricsComputer import")
        except ImportError as e:
            self.results.fail_test("MetricsComputer import", str(e))
    
    def test_callbacks_import(self):
        """Test training callbacks can be imported."""
        try:
            from finetuning.finetuners.training.callbacks import (
                SavePeftModelCallback, EpochMetricsCallback
            )
            self.results.pass_test("Callbacks import")
        except ImportError as e:
            self.results.fail_test("Callbacks import", str(e))
    
    def run_all(self):
        """Run all training module tests."""
        print("\n🧪 Training Module Tests")
        print("-" * 40)
        
        self.test_trainer_import()
        self.test_metrics_import()
        self.test_callbacks_import()


class TestMainFinetuner:
    """Tests for main finetuner class."""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def test_finetuner_import(self):
        """Test SmolLM2VBLoRAFineTuner can be imported."""
        try:
            from finetuning.finetuners.finetuning import SmolLM2VBLoRAFineTuner
            self.results.pass_test("SmolLM2VBLoRAFineTuner import")
        except ImportError as e:
            self.results.fail_test("SmolLM2VBLoRAFineTuner import", str(e))
    
    def test_split_function_import(self):
        """Test split_synthetic_data can be imported."""
        try:
            from finetuning.finetuners.finetuning import split_synthetic_data
            self.results.pass_test("split_synthetic_data import")
        except ImportError as e:
            self.results.fail_test("split_synthetic_data import", str(e))
    
    def test_finetuner_init(self):
        """Test finetuner initialization."""
        try:
            from finetuning.finetuners.finetuning import SmolLM2VBLoRAFineTuner
            
            with tempfile.TemporaryDirectory() as temp_dir:
                finetuner = SmolLM2VBLoRAFineTuner(
                    model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
                    output_dir=temp_dir
                )
                
                # Check attributes exist
                checks = [
                    hasattr(finetuner, 'model_name'),
                    hasattr(finetuner, 'output_dir'),
                    hasattr(finetuner, 'data_config'),
                    hasattr(finetuner, 'training_config'),
                    hasattr(finetuner, 'vblora_config'),
                    hasattr(finetuner, 'hardware_config'),
                ]
                
                if all(checks):
                    self.results.pass_test("Finetuner initialization")
                else:
                    self.results.fail_test("Finetuner initialization", "Missing attributes")
        except Exception as e:
            self.results.fail_test("Finetuner initialization", str(e))
    
    def run_all(self):
        """Run all main finetuner tests."""
        print("\n🧪 Main Finetuner Tests")
        print("-" * 40)
        
        self.test_finetuner_import()
        self.test_split_function_import()
        self.test_finetuner_init()


def main():
    """Run all comprehensive tests."""
    print("=" * 60)
    print("🧪 COMPREHENSIVE CLI AND MODULE TEST SUITE")
    print("=" * 60)
    
    results = TestResults()
    
    # Run all test classes
    TestVBLoRACLI(results).run_all()
    TestORPOCLI(results).run_all()
    TestConfigModule(results).run_all()
    TestUtilsModule(results).run_all()
    TestORPOModule(results).run_all()
    TestCoreModule(results).run_all()
    TestTrainingModule(results).run_all()
    TestMainFinetuner(results).run_all()
    
    # Print summary
    success = results.summary()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
