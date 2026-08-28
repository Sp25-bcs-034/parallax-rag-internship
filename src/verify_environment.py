"""
verify_environment.py

Week 1 - Environment Verification Script

"""

import sys
from importlib.metadata import version
import pandas as pd
import requests
import pytest
import chardet
from dotenv import load_dotenv



def check_python():
    """Display Python version."""
    print("\n[Python]")
    print(f"✓ Python Version : {sys.version.split()[0]}")


def check_libraries():
    """Verify required libraries."""
    print("\n[Installed Libraries]")

    print(f"✓ pandas          : {pd.__version__}")
    print(f"✓ requests        : {version('requests')}")
    print(f"✓ pytest          : {pytest.__version__}")
    print(f"✓ chardet         : {chardet.__version__}")
    load_dotenv()
    print("✓ python-dotenv  : Installed")


def main():
    print("=" * 60)
    print(" Parallax Labs - Week 1 Environment Verification")
    check_python()
    check_libraries()

    print("\n" + "=" * 60)
    print(" Environment verification completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()