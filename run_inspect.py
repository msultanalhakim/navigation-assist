"""
Runner for model inspection
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import argparse

# Import after path setup
from tools.inspect_model import inspect_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect YOLOv11 Model')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (.pt)')
    
    args = parser.parse_args()
    
    if args.model:
        inspect_model(args.model)
    else:
        print("Usage: python run_inspect.py --model path/to/model.pt")
        print("Or set project/experiment in config.yaml to use latest model")