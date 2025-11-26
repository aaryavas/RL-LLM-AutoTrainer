#!/usr/bin/env python3
"""
Master test runner for all finetuning tests.
Runs comprehensive CLI tests, integration tests, and unit tests.
"""

import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime

# Add parent directories to path
_current_dir = Path(__file__).parent.resolve()
_finetuners_dir = _current_dir.parent
_finetuning_dir = _finetuners_dir.parent
_prototyping_dir = _finetuning_dir.parent

sys.path.insert(0, str(_prototyping_dir))


def run_test_file(test_file: Path) -> tuple:
    """Run a test file and return (success, output)."""
    result = subprocess.run(
        [sys.executable, str(test_file)],
        capture_output=True,
        text=True,
        cwd=str(_prototyping_dir)
    )
    
    output = result.stdout + result.stderr
    success = result.returncode == 0
    
    return success, output


def main():
    """Run all tests."""
    print("=" * 70)
    print("🧪 FINETUNING MODULE - MASTER TEST RUNNER")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    test_files = [
        ("ORPO Generator Tests", _current_dir / "test_orpo_generator.py"),
        ("Comprehensive CLI Tests", _current_dir / "test_cli_comprehensive.py"),
        ("Integration Tests", _current_dir / "test_integration.py"),
    ]
    
    results = []
    
    for name, test_file in test_files:
        print(f"\n{'='*70}")
        print(f"📋 Running: {name}")
        print(f"   File: {test_file.name}")
        print("=" * 70)
        
        if not test_file.exists():
            print(f"   ⚠️ Test file not found: {test_file}")
            results.append((name, False, "File not found"))
            continue
        
        success, output = run_test_file(test_file)
        
        # Print output
        print(output)
        
        if success:
            results.append((name, True, ""))
        else:
            results.append((name, False, "Tests failed"))
    
    # Print final summary
    print("\n" + "=" * 70)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    for name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        error_msg = f" ({error})" if error else ""
        print(f"   {status}: {name}{error_msg}")
    
    print()
    print(f"   Total: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
