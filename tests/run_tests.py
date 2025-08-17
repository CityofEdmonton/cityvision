#!/usr/bin/env python3
"""
Test runner script for the train.py module.
This script provides an easy way to run all tests or specific test classes.
"""

import pytest
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests():
    """Run all tests in the test_train.py file."""
    print("Running all tests for train.py...")
    pytest.main(["-v", "test_train.py"])


def run_specific_test_class(test_class_name):
    """Run tests from a specific test class."""
    print(f"Running tests from {test_class_name}...")
    pytest.main(["-v", f"test_train.py::{test_class_name}"])


def run_specific_test(test_name):
    """Run a specific test by name."""
    print(f"Running test: {test_name}...")
    pytest.main(["-v", f"test_train.py::{test_name}"])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            run_all_tests()
        elif sys.argv[1] == "--class" and len(sys.argv) > 2:
            run_specific_test_class(sys.argv[2])
        elif sys.argv[1] == "--test" and len(sys.argv) > 2:
            run_specific_test(sys.argv[2])
        else:
            print("Usage:")
            print("  python run_tests.py --all                    # Run all tests")
            print("  python run_tests.py --class TestClassName    # Run specific test class")
            print("  python run_tests.py --test test_function_name # Run specific test")
    else:
        run_all_tests()
