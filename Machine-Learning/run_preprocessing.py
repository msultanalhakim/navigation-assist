"""
Runner for preprocessing pipeline
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.preprocessing.optimized_pipeline import main

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Optimized Dataset Processing')
    parser.add_argument('--force', action='store_true', help='Force reprocessing')
    args = parser.parse_args()
    
    main(force=args.force)