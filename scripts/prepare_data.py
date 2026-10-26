"""
Canonical data preparation driver script for the SWaT testbed dataset.
Standardized CLI interface wrapping SWaTLoader.
"""

import sys
import os

# Delegate directly to preprocess_swat main routine
from scripts.preprocess_swat import main

if __name__ == "__main__":
    main()
