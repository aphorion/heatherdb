#!/usr/bin/env python3
"""Run the Cipher CLI."""

import sys
import os

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(__file__))
from cipher.cli import run_cli

if __name__ == "__main__":
    run_cli()
