"""
Runner for model evaluation
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.training.evaluate_model import main

if __name__ == "__main__":
    main()