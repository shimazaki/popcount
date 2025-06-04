#!/usr/bin/env python
"""
Test runner for the population count model project

Author: Hideaki Shimazaki
"""

import sys
import pytest

def main():
    """Run all tests with coverage reporting"""
    args = [
        # Test discovery
        ".",
        # Verbose output
        "-v",
        # Show local variables in tracebacks
        "-l",
        # Stop on first failure (optional, remove if you want to see all failures)
        "-x",
        # Show stdout/stderr
        "-s",
        # Pattern for test files
        "--tb=short",
        # Coverage report (requires pytest-cov)
        "--cov=.",
        "--cov-report=term-missing",
        "--cov-report=html",
        # Exclude test files from coverage
        "--cov-config=.coveragerc"
    ]
    
    # Run pytest with the specified arguments
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\nAll tests passed! ✅")
    else:
        print("\nSome tests failed. ❌")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())