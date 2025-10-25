"""
Script to inspect YOLOv11 model input/output shapes
"""

from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image
import argparse

from scripts.utils import logger, config


def inspect_model(model_path: str):
    """Inspect model architecture and shapes"""
    
    logger.info("=" * 80)
    logger.info("YOLO MODEL INSPECTION")
    logger.info("=" * 80)
    
    # Load model
    try:
        model = YOLO(model_path)
        logger.info(f"\nModel loaded: {model_path}\n")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Method 1: Model Architecture
    logger.info("=" * 80)
    logger.info("METHOD 1: Model Architecture")
    logger.info("-" * 80)
    print(model.model)
    print()
    
    # Method 2: Dummy Input Test
    logger.info("=" * 80)
    logger.info("METHOD 2: Dummy Input Test")
    logger.info("-" * 80)
    
    dummy_input = torch.randn(1, 3, 640, 640)
    logger.info(f"Input Shape: {dummy_input.shape}")
    logger.info(f"  Format: [batch_size, channels, height, width]")
    logger.info(f"  Values: [{dummy_input.shape[0]}, {dummy_input.shape[1]}, "
               f"{dummy_input.shape[2]}, {dummy_input.shape[3]}]")
    
    model.model.eval()
    with torch.no_grad():
        output = model.model(dummy_input)
    
    logger.info(f"\nOutput Type: {type(output)}")
    
    if isinstance(output, (tuple, list)):
        logger.info(f"Number of outputs: {len(output)}")
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor):
                logger.info(f"\nOutput[{i}] Shape: {out.shape}")
                logger.info(f"  Total elements: {out.numel()}")
    else:
        logger.info(f"Output Shape: {output.shape}")
    
    print()
    
    # Method 3: Model Info
    logger.info("=" * 80)
    logger.info("METHOD 3: Model Info")
    logger.info("-" * 80)
    model.info(detailed=False, verbose=True)
    print()
    
    # Method 4: Real Prediction Test
    logger.info("=" * 80)
    logger.info("METHOD 4: Real Prediction Test")
    logger.info("-" * 80)
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    dummy_image_pil = Image.fromarray(dummy_image)
    
    # Predict
    results = model.predict(dummy_image_pil, verbose=False)
    
    logger.info(f"Input: PIL Image or numpy array (H, W, C)")
    logger.info(f"  Example shape: (640, 640, 3)")
    logger.info(f"\nOutput: Results object")
    logger.info(f"  Type: {type(results[0])}")
    logger.info(f"  Boxes: {results[0].boxes.shape if results[0].boxes is not None else 'None'}")
    logger.info(f"  Number of detections: {len(results[0].boxes) if results[0].boxes is not None else 0}")
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        logger.info(f"\nBox format: [x1, y1, x2, y2, confidence, class]")
        logger.info(f"  boxes.xyxy: Bounding box coordinates")
        logger.info(f"  boxes.conf: Confidence scores")
        logger.info(f"  boxes.cls: Class indices")
    
    print()
    
    # Summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"""
INPUT SHAPE:
  - Tensor format: torch.Tensor([1, 3, 640, 640])
  - Image format: numpy.ndarray (H, W, C) or PIL.Image
  - Dimensions: 640x640 pixels (default, configurable)
  - Channels: 3 (RGB)

OUTPUT SHAPE:
  - Detection boxes: [N, 6] where N = number of detections
  - Format per box: [x1, y1, x2, y2, confidence, class_id]
  - Class names: {model.names}

USAGE EXAMPLE:
  results = model.predict("image.jpg")
  boxes = results[0].boxes.xyxy   # Coordinates [x1,y1,x2,y2]
  conf = results[0].boxes.conf     # Confidence scores
  cls = results[0].boxes.cls       # Class IDs
    """)
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Inspect YOLOv11 Model')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (.pt)')
    
    args = parser.parse_args()
    
    # Get model path
    if args.model:
        model_path = args.model
    else:
        # Try to get from config
        project = config.get('project', 'runs/train')
        experiment = config.get('experiment', 'exp_latest')
        model_path = f"{project}/{experiment}/weights/best.pt"
    
    inspect_model(model_path)


if __name__ == "__main__":
    main()