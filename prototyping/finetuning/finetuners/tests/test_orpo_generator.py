#!/usr/bin/env python3
"""
Test script for ORPO data generator.
Tests the conversion of synthetic data (text/label) to ORPO format (prompt/chosen/rejected).
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
_current_dir = Path(__file__).parent.resolve()
_finetuners_dir = _current_dir.parent
_finetuning_dir = _finetuners_dir.parent
_prototyping_dir = _finetuning_dir.parent

sys.path.insert(0, str(_prototyping_dir))

import pandas as pd
import tempfile


def test_orpo_generator_without_model():
    """Test ORPO dataset creation without model (using provided rejected responses)."""
    print("=" * 60)
    print("Test 1: ORPO Generator - External Rejected Responses")
    print("=" * 60)
    
    from finetuning.finetuners.utils.orpo_generator import ORPODataGenerator
    
    # Create sample data
    df = pd.DataFrame({
        'text': [
            'What is the capital of France?',
            'Explain machine learning briefly.',
            'What is 2 + 2?',
        ],
        'label': [
            'The capital of France is Paris.',
            'Machine learning is a subset of AI that enables systems to learn from data.',
            'The answer is 4.',
        ]
    })
    
    rejected_df = pd.DataFrame({
        'rejected_response': [
            'France is a country.',
            'ML is about computers.',
            'The answer is 5.',
        ]
    })
    
    # Initialize generator (no model needed for external rejected responses)
    generator = ORPODataGenerator()
    
    # Create ORPO dataset
    orpo_df = generator.create_orpo_dataset(
        df=df,
        rejected_df=rejected_df,
        prompt_col='text',
        chosen_col='label',
        rejected_col='rejected_response'
    )
    
    print(f"\n✅ Created ORPO dataset with {len(orpo_df)} examples")
    print("\nSample data:")
    for i, row in orpo_df.head(2).iterrows():
        print(f"\n  Example {i+1}:")
        print(f"    Prompt: {row['prompt'][:50]}...")
        print(f"    Chosen: {row['chosen'][:50]}...")
        print(f"    Rejected: {row['rejected'][:50]}...")
    
    # Validate output format
    assert 'prompt' in orpo_df.columns, "Missing 'prompt' column"
    assert 'chosen' in orpo_df.columns, "Missing 'chosen' column"
    assert 'rejected' in orpo_df.columns, "Missing 'rejected' column"
    assert len(orpo_df) == 3, f"Expected 3 rows, got {len(orpo_df)}"
    
    print("\n✅ Test 1 PASSED: External rejected responses work correctly")
    return True


def test_orpo_generator_from_synthetic():
    """Test ORPO dataset creation from synthetic data format (without model)."""
    print("\n" + "=" * 60)
    print("Test 2: ORPO Generator - From Synthetic Data Format")
    print("=" * 60)
    
    from finetuning.finetuners.utils.orpo_generator import ORPODataGenerator
    
    # Create sample synthetic data
    df = pd.DataFrame({
        'text': [
            'Classify this sentiment: Great product!',
            'Classify this sentiment: Terrible experience.',
            'Classify this sentiment: It was okay.',
        ],
        'label': [
            'positive',
            'negative',
            'neutral',
        ],
        'model': ['test-model'] * 3,
        'reasoning': ['test reasoning'] * 3,
    })
    
    # Initialize generator without model (generate_rejected=False)
    generator = ORPODataGenerator()
    
    # Test that we can't generate without a model
    try:
        orpo_df = generator.create_orpo_dataset_from_synthetic(
            df=df,
            text_column='text',
            label_column='label',
            generate_rejected=True,  # This should fail without a model
        )
        print("❌ Should have raised RuntimeError")
        return False
    except RuntimeError as e:
        print(f"✅ Correctly raised RuntimeError: {e}")
    
    # Test with generate_rejected=False (requires external handling)
    orpo_df = generator.create_orpo_dataset_from_synthetic(
        df=df,
        text_column='text',
        label_column='label',
        generate_rejected=False,  # Don't generate - just placeholder
    )
    
    # This will have empty rejected responses
    print(f"\n  Created dataset with {len(orpo_df)} examples (no rejected yet)")
    
    print("\n✅ Test 2 PASSED: Synthetic data format conversion works")
    return True


def test_orpo_generator_save_to_file():
    """Test saving ORPO dataset to CSV file."""
    print("\n" + "=" * 60)
    print("Test 3: ORPO Generator - Save to File")
    print("=" * 60)
    
    from finetuning.finetuners.utils.orpo_generator import ORPODataGenerator
    
    # Create sample data
    df = pd.DataFrame({
        'text': ['Question 1', 'Question 2'],
        'label': ['Answer 1', 'Answer 2'],
    })
    
    rejected_df = pd.DataFrame({
        'rejected_response': ['Wrong 1', 'Wrong 2'],
    })
    
    generator = ORPODataGenerator()
    
    # Create ORPO dataset
    orpo_df = generator.create_orpo_dataset(
        df=df,
        rejected_df=rejected_df,
        prompt_col='text',
        chosen_col='label',
        rejected_col='rejected_response'
    )
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        orpo_df.to_csv(f.name, index=False)
        temp_path = f.name
    
    # Verify file
    loaded_df = pd.read_csv(temp_path)
    assert len(loaded_df) == 2, f"Expected 2 rows, got {len(loaded_df)}"
    assert list(loaded_df.columns) == ['prompt', 'chosen', 'rejected']
    
    # Cleanup
    os.unlink(temp_path)
    
    print("✅ Test 3 PASSED: File saving works correctly")
    return True


def test_merge_adapter_utility():
    """Test the merge adapter utility functions."""
    print("\n" + "=" * 60)
    print("Test 4: Merge Adapter Utility")
    print("=" * 60)
    
    from finetuning.finetuners.utils.merge_adapter import get_base_model_from_adapter
    
    # Test with non-existent path (should return None)
    result = get_base_model_from_adapter("/non/existent/path")
    assert result is None, "Should return None for non-existent path"
    print("✅ get_base_model_from_adapter returns None for invalid path")
    
    print("\n✅ Test 4 PASSED: Merge adapter utility works correctly")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 ORPO Generator Test Suite")
    print("=" * 60)
    
    tests = [
        test_orpo_generator_without_model,
        test_orpo_generator_from_synthetic,
        test_orpo_generator_save_to_file,
        test_merge_adapter_utility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
