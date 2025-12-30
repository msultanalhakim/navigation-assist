"""
Runner for model export
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.export.export_tflite import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export model to TFLite')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (.pt)')
    
    args = parser.parse_args()
    main(args.model)