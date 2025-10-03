#!/usr/bin/env python3
"""
Convenience script to regenerate AI descriptions
"""

import sys
import os

# Add the current directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.scripts.regenerate_descriptions import main

if __name__ == "__main__":
    main()
