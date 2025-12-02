#!/usr/bin/env python3
"""
Comprehensive ORPO Pipeline Test
Tests the complete ORPO workflow without requiring GPU or actual training.
"""

import sys
import os
import subprocess
import tempfile
import json
from pathlib import Path

# Add paths
_current_dir = Path(__file__).parent.resolve()
_finetuners_dir = _current_dir.parent
_finetuning_dir = _finetuners_dir.parent
_prototyping_dir = _finetuning_dir.parent

sys.path.insert(0, str(_prototyping_dir))

import pandas as pd


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_test(name: str, passed: bool, detail: str = ""):
    """Print test result."""
    status = "✅" if passed else "❌"
    print(f"  {status} {name}")
    if detail and not passed:
        print(f"      → {detail}")


def run_command(args: list, cwd: str = None) -> tuple:
    """Run a command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=cwd or str(_prototyping_dir)
    )
    return result.returncode, result.stdout, result.stderr


def create_test_synthetic_data() -> str:
    """Create test synthetic data file."""
    data = pd.DataFrame({
        'text': [
            'What is the capital of France?',
            'Explain photosynthesis briefly.',
            'What is machine learning?',
            'Describe the water cycle.',
            'What is DNA?',
            'Define gravity.',
            'What causes earthquakes?',
            'Explain evolution.',
            'What is a black hole?',
            'Describe the immune system.',
            'What is climate change?',
            'Define biodiversity.',
            'Explain the scientific method.',
            'What is an ecosystem?',
            'Describe cellular respiration.',
        ],
        'label': [
            'The capital of France is Paris, known for the Eiffel Tower.',
            'Photosynthesis converts sunlight into energy using chlorophyll.',
            'Machine learning is a subset of AI that enables systems to learn from data.',
            'The water cycle involves evaporation, condensation, and precipitation.',
            'DNA is a molecule that carries genetic instructions for living organisms.',
            'Gravity is the force that attracts objects with mass toward each other.',
            'Earthquakes are caused by sudden release of energy in Earth\'s crust.',
            'Evolution is the process by which species change through natural selection.',
            'A black hole is a region where gravity is so strong nothing can escape.',
            'The immune system defends the body against pathogens.',
            'Climate change refers to long-term shifts in temperatures and weather.',
            'Biodiversity is the variety of life forms in an ecosystem.',
            'The scientific method involves observation, hypothesis, and experimentation.',
            'An ecosystem is a community of organisms interacting with their environment.',
            'Cellular respiration converts glucose and oxygen into energy.',
        ]
    })
    
    path = tempfile.mktemp(suffix='.csv')
    data.to_csv(path, index=False)
    return path


def create_test_orpo_data() -> str:
    """Create test ORPO data file."""
    data = pd.DataFrame({
        'prompt': [
            'What is the capital of France?',
            'Explain photosynthesis briefly.',
            'What is machine learning?',
            'Describe the water cycle.',
            'What is DNA?',
            'Define gravity.',
            'What causes earthquakes?',
            'Explain evolution.',
            'What is a black hole?',
            'Describe the immune system.',
        ],
        'chosen': [
            'The capital of France is Paris, known for the Eiffel Tower.',
            'Photosynthesis converts sunlight into energy using chlorophyll.',
            'Machine learning is a subset of AI that enables systems to learn from data.',
            'The water cycle involves evaporation, condensation, and precipitation.',
            'DNA is a molecule that carries genetic instructions for living organisms.',
            'Gravity is the force that attracts objects with mass toward each other.',
            'Earthquakes are caused by sudden release of energy in Earth\'s crust.',
            'Evolution is the process by which species change through natural selection.',
            'A black hole is a region where gravity is so strong nothing can escape.',
            'The immune system defends the body against pathogens.',
        ],
        'rejected': [
            'France is a country in Europe.',
            'Plants need sunlight to grow.',
            'Computers can do calculations quickly.',
            'Water is important for life.',
            'DNA is found in cells.',
            'Things fall down when dropped.',
            'The ground shakes sometimes.',
            'Animals change over time.',
            'Space has many objects.',
            'The body fights germs.',
        ]
    })
    
    path = tempfile.mktemp(suffix='.csv')
    data.to_csv(path, index=False)
    return path


class TestORPOGenerator:
    """Test ORPO data generator."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def run(self):
        print_header("ORPO Data Generator Tests")
        
        from finetuning.finetuners.utils.orpo_generator import ORPODataGenerator
        
        # Test 1: Basic initialization
        try:
            generator = ORPODataGenerator()
            print_test("Initialization without model", True)
            self.passed += 1
        except Exception as e:
            print_test("Initialization without model", False, str(e))
            self.failed += 1
        
        # Test 2: Create ORPO dataset with external rejected
        try:
            df = pd.DataFrame({
                'text': ['Question 1', 'Question 2', 'Question 3'],
                'label': ['Answer 1', 'Answer 2', 'Answer 3']
            })
            rejected_df = pd.DataFrame({
                'rejected_response': ['Bad 1', 'Bad 2', 'Bad 3']
            })
            
            result = generator.create_orpo_dataset(df, rejected_df)
            
            assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
            assert 'prompt' in result.columns
            assert 'chosen' in result.columns
            assert 'rejected' in result.columns
            
            print_test("Create ORPO dataset (external rejected)", True)
            self.passed += 1
        except Exception as e:
            print_test("Create ORPO dataset (external rejected)", False, str(e))
            self.failed += 1
        
        # Test 3: From synthetic format (no model - should fail gracefully)
        try:
            synthetic_df = pd.DataFrame({
                'text': ['Q1', 'Q2'],
                'label': ['A1', 'A2']
            })
            
            try:
                generator.create_orpo_dataset_from_synthetic(
                    synthetic_df, generate_rejected=True
                )
                print_test("From synthetic (no model) raises error", False, "Should have raised")
                self.failed += 1
            except RuntimeError:
                print_test("From synthetic (no model) raises error", True)
                self.passed += 1
        except Exception as e:
            print_test("From synthetic (no model) raises error", False, str(e))
            self.failed += 1
        
        # Test 4: Save to file
        try:
            df = pd.DataFrame({
                'text': ['Q1'],
                'label': ['A1']
            })
            rejected_df = pd.DataFrame({
                'rejected_response': ['R1']
            })
            result = generator.create_orpo_dataset(df, rejected_df)
            
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
                result.to_csv(f.name, index=False)
                loaded = pd.read_csv(f.name)
                os.unlink(f.name)
            
            assert len(loaded) == len(result)
            print_test("Save and load ORPO dataset", True)
            self.passed += 1
        except Exception as e:
            print_test("Save and load ORPO dataset", False, str(e))
            self.failed += 1


class TestORPOCLI:
    """Test ORPO CLI."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.cli_path = str(_finetuners_dir / "orpo" / "cli.py")
    
    def run(self):
        print_header("ORPO CLI Tests")
        
        # Test 1: Help command
        code, stdout, stderr = run_command(["python", self.cli_path, "--help"])
        if code == 0 and "ORPO" in stdout:
            print_test("--help command", True)
            self.passed += 1
        else:
            print_test("--help command", False, f"Exit: {code}")
            self.failed += 1
        
        # Test 2: All required arguments present
        code, stdout, stderr = run_command(["python", self.cli_path, "--help"])
        required_args = ["--model_name", "--data_path", "--beta", "--dry-run"]
        missing = [arg for arg in required_args if arg not in stdout]
        if not missing:
            print_test("All required arguments present", True)
            self.passed += 1
        else:
            print_test("All required arguments present", False, f"Missing: {missing}")
            self.failed += 1
        
        # Test 3: Dry run with valid data
        orpo_data_path = create_test_orpo_data()
        try:
            code, stdout, stderr = run_command([
                "python", self.cli_path,
                "--data_path", orpo_data_path,
                "--dry-run"
            ])
            if code == 0 and "Dry run mode" in stdout:
                print_test("Dry run with valid ORPO data", True)
                self.passed += 1
            else:
                print_test("Dry run with valid ORPO data", False, f"Exit: {code}, Out: {stdout[:200]}")
                self.failed += 1
        finally:
            os.unlink(orpo_data_path)
        
        # Test 4: Missing file error
        code, stdout, stderr = run_command([
            "python", self.cli_path,
            "--data_path", "/nonexistent/file.csv"
        ])
        if code != 0 and "not found" in (stdout + stderr).lower():
            print_test("Missing file error handling", True)
            self.passed += 1
        else:
            print_test("Missing file error handling", False, "No proper error")
            self.failed += 1
        
        # Test 5: Custom parameters in dry run
        orpo_data_path = create_test_orpo_data()
        try:
            code, stdout, stderr = run_command([
                "python", self.cli_path,
                "--data_path", orpo_data_path,
                "--beta", "0.2",
                "--epochs", "5",
                "--lr", "2e-5",
                "--dry-run"
            ])
            if code == 0 and "0.2" in stdout and "5" in stdout:
                print_test("Custom parameters accepted", True)
                self.passed += 1
            else:
                print_test("Custom parameters accepted", False, f"Exit: {code}")
                self.failed += 1
        finally:
            os.unlink(orpo_data_path)


class TestVBLoRACLIWithORPO:
    """Test VB-LoRA CLI with ORPO options."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.cli_path = str(_finetuners_dir / "cli.py")
    
    def run(self):
        print_header("VB-LoRA CLI + ORPO Integration Tests")
        
        # Test 1: ORPO options present in help
        code, stdout, stderr = run_command([
            "python", self.cli_path, "finetune", "--help"
        ])
        orpo_args = ["--orpo", "--orpo-epochs", "--orpo-beta", "--orpo-lr"]
        found = all(arg in stdout for arg in orpo_args)
        if found:
            print_test("ORPO options in finetune help", True)
            self.passed += 1
        else:
            print_test("ORPO options in finetune help", False, "Missing ORPO args")
            self.failed += 1
        
        # Test 2: Dry run with --orpo flag
        synthetic_data_path = create_test_synthetic_data()
        try:
            code, stdout, stderr = run_command([
                "python", self.cli_path, "finetune", synthetic_data_path,
                "--variant", "SmolLM2-135M",
                "--dry-run",
                "--orpo"
            ])
            if code == 0 and "Dry run" in stdout:
                print_test("Dry run with --orpo flag", True)
                self.passed += 1
            else:
                print_test("Dry run with --orpo flag", False, f"Exit: {code}")
                self.failed += 1
        finally:
            os.unlink(synthetic_data_path)
        
        # Test 3: ORPO parameters passed correctly
        synthetic_data_path = create_test_synthetic_data()
        try:
            code, stdout, stderr = run_command([
                "python", self.cli_path, "finetune", synthetic_data_path,
                "--variant", "SmolLM2-135M",
                "--dry-run",
                "--orpo",
                "--orpo-epochs", "5",
                "--orpo-beta", "0.2",
                "--orpo-lr", "2e-5"
            ])
            if code == 0:
                print_test("Custom ORPO parameters", True)
                self.passed += 1
            else:
                print_test("Custom ORPO parameters", False, f"Exit: {code}")
                self.failed += 1
        finally:
            os.unlink(synthetic_data_path)


class TestORPOConfig:
    """Test ORPO configuration."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def run(self):
        print_header("ORPO Configuration Tests")
        
        # Test 1: Config import
        try:
            from finetuning.finetuners.orpo.config.orpo_config import ORPOSpecificConfig
            print_test("ORPOSpecificConfig import", True)
            self.passed += 1
        except ImportError as e:
            print_test("ORPOSpecificConfig import", False, str(e))
            self.failed += 1
            return
        
        # Test 2: Default values
        config = ORPOSpecificConfig()
        checks = [
            ("beta", config.beta == 0.5),
            ("max_prompt_length", config.max_prompt_length == 512),
            ("max_completion_length", config.max_completion_length == 1024),
            ("disable_dropout", config.disable_dropout == True),
        ]
        
        for name, passed in checks:
            print_test(f"Default {name}", passed)
            if passed:
                self.passed += 1
            else:
                self.failed += 1


class TestORPOTrainer:
    """Test ORPO trainer imports."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def run(self):
        print_header("ORPO Trainer Tests")
        
        # Test 1: Trainer import
        try:
            from finetuning.finetuners.orpo.train_orpo import ORPOFineTuner
            print_test("ORPOFineTuner import", True)
            self.passed += 1
        except ImportError as e:
            print_test("ORPOFineTuner import", False, str(e))
            self.failed += 1
        
        # Test 2: Data processor import
        try:
            from finetuning.finetuners.orpo.core.data_processor import PreferenceDataProcessor
            print_test("PreferenceDataProcessor import", True)
            self.passed += 1
        except ImportError as e:
            print_test("PreferenceDataProcessor import", False, str(e))
            self.failed += 1
        
        # Test 3: Visualization import
        try:
            from finetuning.finetuners.orpo.training.visualization import TrainingVisualizer
            print_test("TrainingVisualizer import", True)
            self.passed += 1
        except ImportError as e:
            print_test("TrainingVisualizer import", False, str(e))
            self.failed += 1


class TestMergeAdapter:
    """Test merge adapter utility."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def run(self):
        print_header("Merge Adapter Utility Tests")
        
        # Test 1: Import
        try:
            from finetuning.finetuners.utils.merge_adapter import (
                merge_adapter, merge_for_orpo, get_base_model_from_adapter
            )
            print_test("Merge adapter functions import", True)
            self.passed += 1
        except ImportError as e:
            print_test("Merge adapter functions import", False, str(e))
            self.failed += 1
            return
        
        # Test 2: get_base_model_from_adapter with nonexistent path
        result = get_base_model_from_adapter("/nonexistent/path")
        if result is None:
            print_test("get_base_model handles missing path", True)
            self.passed += 1
        else:
            print_test("get_base_model handles missing path", False, f"Got: {result}")
            self.failed += 1
        
        # Test 3: merge_for_orpo with invalid path
        try:
            merge_for_orpo("/nonexistent/adapter")
            print_test("merge_for_orpo error handling", False, "Should have raised")
            self.failed += 1
        except Exception:
            print_test("merge_for_orpo error handling", True)
            self.passed += 1


def main():
    """Run all ORPO pipeline tests."""
    print("\n" + "="*60)
    print("  COMPREHENSIVE ORPO PIPELINE TEST SUITE")
    print("  Testing: Generator, CLI, Config, Trainer, Merge Adapter")
    print("="*60)
    
    total_passed = 0
    total_failed = 0
    
    # Run all test classes
    test_classes = [
        TestORPOGenerator(),
        TestORPOCLI(),
        TestVBLoRACLIWithORPO(),
        TestORPOConfig(),
        TestORPOTrainer(),
        TestMergeAdapter(),
    ]
    
    for test_class in test_classes:
        test_class.run()
        total_passed += test_class.passed
        total_failed += test_class.failed
    
    # Summary
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    print(f"  Total Passed: {total_passed}")
    print(f"  Total Failed: {total_failed}")
    print(f"  Success Rate: {total_passed / (total_passed + total_failed) * 100:.1f}%")
    print("="*60)
    
    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
