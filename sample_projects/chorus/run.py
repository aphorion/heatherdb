#!/usr/bin/env python3
"""Run the Chorus CLI."""

import sys
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(__file__))
from chorus.cli import run_cli

if __name__ == "__main__":
    run_cli()
