#!/usr/bin/env python3
"""
Selective model test - SELECT IMAGES BY FILENAME
Edit the SELECTED_IMAGES list below!
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
from pathlib import Path
import json

# ============================================================================
# 🎯 EDIT THIS SECTION - SELECT YOUR IMAGES HERE!
# ============================================================================

SELECTED_IMAGES = [
    # PERSON images (2)
    "10-129.jpg",
    "WhatsApp-Image-2025-09-29-at-17_28_23-1-_jpeg.rf.11e35fb28317800c5b0a3bc19bf6fbe1.jpg",
    
    # CHAIR images (2)
    "cadeira-de-escritorio-azul_mp4-56.jpg",
    "door-7.jpg",  # This should be a chair image despite the name
    
    # TABLE images (2)
    "images-27_dup1.jpg",
    "img785281761240.jpg",
    
    # DOOR images (2)
    "door-76.jpg",  # Replace with actual door images!
    "WhatsApp-Image-2025-10-10-at-19_30_24-12-_jpeg.rf.2b790aee85d9c76777af8098d4de3434.jpg",  # Replace with actual door images!
    
    # STAIR images (2)
    "WhatsApp-Image-2025-10-10-at-19_30_26-8-_jpeg.rf.e722cf16f04d7364dff7b1a858ec0329.jpg",
    "WhatsApp-Image-2025-10-10-at-19_30_26-1-_jpeg.rf.a49c8ad5b2536b3be3bd1e38ada1e807.jpg",
]

# ============================================================================
# CONFIG
# ============================================================================

MODEL_PATH = "yolov11.tflite"
DATASET_PATH = Path("dataset") / "processed" / "test" / "images"
OUTPUT_JSON = "selective_test_results.json"
CLASS_NAMES = ["person", "chair", "table", "door", "stair"]
CONFIDENCE_THRESHOLD = 0.5

# ============================================================================
# SETUP
# ============================================================================

def load_interpreter():
    """Load TFLite interpreter"""
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def find_image(filename: str) -> Path:
    """Find image by filename in dataset"""
    # Try exact match first
    img_path = DATASET_PATH / filename
    if img_path.exists():
        return img_path
    
    # Try with different extensions
    for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
        img_path = DATASET_PATH / (Path(filename).stem + ext)
        if img_path.exists():
            return img_path
    
    return None


def preprocess_image(image_path: str, input_size: int = 640) -> np.ndarray:
    """Preprocess image exactly like Android"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        img = cv2.resize(img, (input_size, input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"  ERROR preprocessing: {e}")
        return None


def run_inference(interpreter, img_batch):
    """Run inference"""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        return output
    except Exception as e:
        print(f"  ERROR inference: {e}")
        return None


def analyze_output(output):
    """Analyze predictions"""
    
    output = output[0]  # Remove batch dim
    
    detections = []
    
    for pred_idx in range(output.shape[1]):
        max_prob = 0
        max_class = 0
        
        for class_idx in range(5):
            prob = output[4 + class_idx][pred_idx]
            if prob > max_prob:
                max_prob = prob
                max_class = class_idx
        
        if max_prob >= CONFIDENCE_THRESHOLD:
            detections.append({
                'class': CLASS_NAMES[max_class],
                'class_id': max_class,
                'confidence': float(max_prob),
                'x': float(output[0][pred_idx]),
                'y': float(output[1][pred_idx]),
                'w': float(output[2][pred_idx]),
                'h': float(output[3][pred_idx])
            })
    
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return detections[:10]


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("SELECTIVE MODEL TEST - BY FILENAME")
    print("=" * 80)
    
    # Check dataset
    if not DATASET_PATH.exists():
        print(f"❌ ERROR: Dataset not found: {DATASET_PATH}")
        return
    
    print(f"\n📁 Dataset path: {DATASET_PATH}")
    print(f"🎯 Selected images: {len(SELECTED_IMAGES)}")
    print()
    
    # Validate all images exist
    print("=" * 80)
    print("VALIDATING SELECTED IMAGES")
    print("=" * 80 + "\n")
    
    valid_images = []
    missing_images = []
    
    for i, filename in enumerate(SELECTED_IMAGES, 1):
        img_path = find_image(filename)
        if img_path:
            print(f"  ✓ [{i:2d}] {filename}")
            valid_images.append((filename, img_path))
        else:
            print(f"  ❌ [{i:2d}] {filename} - NOT FOUND")
            missing_images.append(filename)
    
    if missing_images:
        print(f"\n❌ ERROR: {len(missing_images)} image(s) not found!")
        print("\nMissing files:")
        for fname in missing_images:
            print(f"  - {fname}")
        print("\nPlease check the filenames in SELECTED_IMAGES variable.")
        return
    
    print(f"\n✓ All {len(valid_images)} images found!")
    
    # Load interpreter
    print("\n🔄 Loading model...")
    interpreter = load_interpreter()
    print("✓ Model loaded")
    
    # Process selected images
    print("\n" + "=" * 80)
    print("TESTING SELECTED IMAGES")
    print("=" * 80 + "\n")
    
    all_results = []
    class_detections = {cls: 0 for cls in CLASS_NAMES}
    
    for idx, (filename, image_path) in enumerate(valid_images, 1):
        print(f"[{idx}/{len(valid_images)}] {filename}")
        
        # Preprocess
        img_batch = preprocess_image(image_path)
        if img_batch is None:
            print(f"  ❌ SKIP: Could not load")
            continue
        
        # Inference
        output = run_inference(interpreter, img_batch)
        if output is None:
            print(f"  ❌ SKIP: Inference failed")
            continue
        
        # Analyze
        detections = analyze_output(output)
        
        # Record result
        all_results.append({
            'filename': filename,
            'path': str(image_path),
            'num_detections': len(detections),
            'detections': detections
        })
        
        # Print results
        if detections:
            for i, det in enumerate(detections):
                class_detections[det['class']] += 1
                marker = "→" if i == 0 else " "
                print(f"  {marker} {det['class']:10s} conf={det['confidence']:.4f}")
        else:
            print(f"  ⚠️  No detections above {CONFIDENCE_THRESHOLD}")
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTested: {len(all_results)} images\n")
    
    print("Total detections by class:")
    for cls_name in CLASS_NAMES:
        count = class_detections[cls_name]
        print(f"  {cls_name:10s}: {count:3d}")
    
    print("\n" + "-" * 80)
    print("ANALYSIS:")
    print("-" * 80)
    
    # Expected: 2 detections per class (roughly)
    expected_per_class = 2
    
    for cls_name in CLASS_NAMES:
        count = class_detections[cls_name]
        if count == 0:
            status = "❌ NO DETECTIONS"
        elif count < expected_per_class:
            status = f"⚠️  LOW ({count}/{expected_per_class})"
        elif count == expected_per_class:
            status = f"✓ GOOD ({count}/{expected_per_class})"
        else:
            status = f"✓ HIGH ({count}/{expected_per_class})"
        
        print(f"  {cls_name:10s}: {status}")
    
    # Save results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump({
            'config': {
                'model': MODEL_PATH,
                'dataset': str(DATASET_PATH),
                'confidence_threshold': CONFIDENCE_THRESHOLD,
                'classes': CLASS_NAMES,
                'selected_images': SELECTED_IMAGES
            },
            'results': all_results,
            'summary': {
                'total_tested': len(all_results),
                'class_counts': class_detections,
                'expected_per_class': expected_per_class
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()