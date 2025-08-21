#!/usr/bin/env python3
"""Command-line scripts for few-shot-exemplar-utils package."""

import subprocess
import sys


def check_all():
    """Run all code quality checks: black, flake8, isort, mypy, and pytest."""
    commands = [
        ["black", "--check", "few_shot_exemplars"],
        ["flake8", "--extend-ignore=E501", "few_shot_exemplars"],
        ["isort", "--check-only", "few_shot_exemplars"],
        ["mypy", "few_shot_exemplars"],
        ["pytest"],
    ]

    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            sys.exit(result.returncode)

    print("All checks passed!")


def fix_all():
    """Auto-fix basic formatting issues with black and isort."""
    commands = [["black", "few_shot_exemplars"], ["isort", "few_shot_exemplars"]]

    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            sys.exit(result.returncode)

    print("Fixed formatting and import order")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "check":
            check_all()
        elif sys.argv[1] == "fix":
            fix_all()
        else:
            print("Usage: python scripts.py [check|fix]")
    else:
        print("Usage: python scripts.py [check|fix]")
