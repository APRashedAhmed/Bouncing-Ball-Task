#!/usr/bin/env python3
"""Test runner for no_change variant validation.

This script provides a convenient way to run all no_change tests with
appropriate output formatting, similar to the standalone validation script.
"""
import pytest
import sys
import argparse
from pathlib import Path


def run_no_change_validation(args):
    """Run all no_change variant tests with detailed output."""
    
    print("=" * 60)
    print("NO_CHANGE VARIANT TEST VALIDATION")
    print("=" * 60)
    print()
    
    # Build pytest arguments
    pytest_args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
    ]
    
    # Add test selection based on what user wants to run
    if args.quick:
        # Run only unit tests (fast)
        pytest_args.extend(["-m", "unit and no_change"])
        print("Running quick unit tests only...")
    elif args.integration:
        # Run integration tests
        pytest_args.extend(["-m", "integration and no_change"])
        print("Running integration tests...")
    elif args.all:
        # Run all tests including slow ones
        pytest_args.extend(["-k", "no_change"])
        print("Running ALL tests (including slow ones)...")
    else:
        # Default: run unit and integration, skip slow
        pytest_args.extend(["-m", "no_change and not slow"])
        print("Running standard tests (skipping slow tests)...")
    
    # Add specific test file pattern
    if args.file:
        pytest_args.append(args.file)
    else:
        # Run all no_change test files
        test_dir = Path(__file__).parent
        pytest_args.extend([
            str(test_dir / "test_no_change_comprehensive.py"),
            str(test_dir / "test_no_change_output_format.py"),
            str(test_dir / "test_no_change_parameters.py"),
            str(test_dir / "test_no_change_integration.py"),
        ])
    
    # Add coverage if requested
    if args.coverage:
        pytest_args.extend([
            "--cov=bouncing_ball_task.controlled_bouncing_ball",
            "--cov-report=term-missing"
        ])
    
    # Add output options
    if args.junit:
        pytest_args.extend(["--junit-xml=test_results.xml"])
    
    if args.html:
        pytest_args.extend(["--html=test_results.html", "--self-contained-html"])
    
    print(f"\nRunning: pytest {' '.join(pytest_args)}")
    print("-" * 60)
    
    # Run pytest
    exit_code = pytest.main(pytest_args)
    
    print("-" * 60)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    return exit_code


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run validation tests for no_change variant"
    )
    
    # Test selection options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run only quick unit tests"
    )
    group.add_argument(
        "--integration", "-i",
        action="store_true",
        help="Run only integration tests"
    )
    group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all tests including slow ones"
    )
    
    # Other options
    parser.add_argument(
        "--file", "-f",
        help="Run specific test file"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--junit",
        action="store_true",
        help="Generate JUnit XML report"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report"
    )
    
    args = parser.parse_args()
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("ERROR: pytest is not installed!")
        print("Please install it with: pip install pytest")
        return 1
    
    # Check for optional dependencies
    if args.coverage:
        try:
            import pytest_cov
        except ImportError:
            print("WARNING: pytest-cov not installed, skipping coverage")
            args.coverage = False
    
    if args.html:
        try:
            import pytest_html
        except ImportError:
            print("WARNING: pytest-html not installed, skipping HTML report")
            args.html = False
    
    return run_no_change_validation(args)


if __name__ == "__main__":
    sys.exit(main())