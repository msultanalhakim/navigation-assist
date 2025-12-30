"""
Runner for hyperparameter search
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.training.hyperparameter_search import main

if __name__ == "__main__":
    main()