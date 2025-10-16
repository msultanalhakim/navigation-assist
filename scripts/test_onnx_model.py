#!/usr/bin/env python3
"""
Test ONNX Model - Verify if ONNX has correct class mapping
This will tell us if the problem happens during .pt→ONNX or ONNX→TFLite
"""

import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path
import json

# ============================================================================
# CONFIG
# ============================================================================

ONNX_MODEL = "models/14-10-2025/best.onnx"
DATASET_PATH = Path("dataset") / "processed" / "test" / "images"
CLASS_NAMES = ["person", "chair", "table", "door", "stair"]
CONFIDENCE_THRESHOLD = 0.5

# Test images
SELECTED_IMAGES = [
    "10-129.jpg",                      # Expected: chair
    "65_dup1.jpg",                     # Expected: chair
    "cadeira-de-escritorio-azul_mp4-56.jpg",  # Expected: chair
    "door-7.jpg",                      # Expected: door
    "door-76.jpg",                     # Expected: door
]

# ============================================================================
# FUNCTIONS
# ============================================================================

def preprocess_image(image_path: str, input_size: int = 640) -> np.ndarray:
    """Preprocess image for ONNX model"""
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def analyze_onnx_output(output, conf_threshold=0.5):
    """Analyze ONNX output - shape: [1, 9, 8400]"""
    
    output = output[0]  # Remove batch dim: [9, 8400]
    
    detections = []
    
    # Iterate through predictions
    for pred_idx in range(output.shape[1]):  # 8400 predictions
        # Get bbox coords
        x = output[0][pred_idx]
        y = output[1][pred_idx]
        w = output[2][pred_idx]
        h = output[3][pred_idx]
        
        # Get class probabilities (indices 4-8 for 5 classes)
        class_probs = output[4:9, pred_idx]
        
        # Find max probability and class
        max_prob = np.max(class_probs)
        max_class = np.argmax(class_probs)
        
        if max_prob >= conf_threshold:
            detections.append({
                'class': CLASS_NAMES[max_class],
                'class_id': int(max_class),
                'confidence': float(max_prob),
                'bbox': [float(x), float(y), float(w), float(h)]
            })
    
    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return detections[:10]


def find_image(filename: str) -> Path:
    """Find image by filename"""
    img_path = DATASET_PATH / filename
    if img_path.exists():
        return img_path
    return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("🧪 TESTING ONNX MODEL")
    print("=" * 80)
    print()
    
    # Check ONNX model exists
    if not Path(ONNX_MODEL).exists():
        print(f"❌ ERROR: ONNX model not found: {ONNX_MODEL}")
        return
    
    print(f"📦 Loading ONNX model: {ONNX_MODEL}")
    
    # Load ONNX model
    try:
        session = ort.InferenceSession(ONNX_MODEL, providers=['CPUExecutionProvider'])
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        
        print(f"✓ Model loaded")
        print(f"  Input: {input_name}, shape: {input_shape}")
        print(f"  Output: {output_name}, shape: {output_shape}")
        print()
        
    except Exception as e:
        print(f"❌ Failed to load ONNX model: {e}")
        return
    
    # Validate images
    print("=" * 80)
    print("VALIDATING TEST IMAGES")
    print("=" * 80)
    print()
    
    valid_images = []
    for filename in SELECTED_IMAGES:
        img_path = find_image(filename)
        if img_path:
            print(f"  ✓ {filename}")
            valid_images.append((filename, img_path))
        else:
            print(f"  ❌ {filename} - NOT FOUND")
    
    if not valid_images:
        print("\n❌ No valid images found!")
        return
    
    print(f"\n✓ Found {len(valid_images)} images")
    print()
    
    # Run inference
    print("=" * 80)
    print("RUNNING INFERENCE WITH ONNX MODEL")
    print("=" * 80)
    print()
    
    onnx_results = []
    class_detections = {cls: 0 for cls in CLASS_NAMES}
    
    for idx, (filename, img_path) in enumerate(valid_images, 1):
        print(f"[{idx}/{len(valid_images)}] {filename}")
        
        # Preprocess
        try:
            img_batch = preprocess_image(img_path)
        except Exception as e:
            print(f"  ❌ Preprocessing failed: {e}")
            continue
        
        # Run inference
        try:
            outputs = session.run([output_name], {input_name: img_batch})
            output = outputs[0]
        except Exception as e:
            print(f"  ❌ Inference failed: {e}")
            continue
        
        # Analyze output
        detections = analyze_onnx_output(output, CONFIDENCE_THRESHOLD)
        
        # Store result
        onnx_results.append({
            'filename': filename,
            'detections': detections
        })
        
        # Print detections
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
    print("SUMMARY - ONNX MODEL RESULTS")
    print("=" * 80)
    print()
    
    print("Detections by class:")
    for cls_name in CLASS_NAMES:
        count = class_detections[cls_name]
        print(f"  {cls_name:10s}: {count:3d}")
    
    print()
    
    # Save results
    with open('onnx_model_test_results.json', 'w') as f:
        json.dump({
            'model': ONNX_MODEL,
            'results': onnx_results,
            'summary': class_detections
        }, f, indent=2)
    
    print("📄 Results saved to: onnx_model_test_results.json")
    print()
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    
    print("=" * 80)
    print("📊 COMPARISON: ONNX vs .pt vs TFLite")
    print("=" * 80)
    print()
    
    # Load .pt results
    pt_results_file = "pt_model_test_results.json"
    if Path(pt_results_file).exists():
        with open(pt_results_file, 'r') as f:
            pt_data = json.load(f)
        
        print("Comparing ONNX vs .pt predictions:\n")
        
        pt_lookup = {r['filename']: r for r in pt_data['results']}
        onnx_lookup = {r['filename']: r for r in onnx_results}
        
        mismatches = 0
        
        for filename in onnx_lookup.keys():
            if filename in pt_lookup:
                pt_top = pt_lookup[filename]['detections'][0] if pt_lookup[filename]['detections'] else None
                onnx_top = onnx_lookup[filename]['detections'][0] if onnx_lookup[filename]['detections'] else None
                
                if pt_top and onnx_top:
                    if pt_top['class'] == onnx_top['class']:
                        print(f"{filename}: ✓ MATCH (both: {pt_top['class']})")
                    else:
                        print(f"{filename}: ❌ MISMATCH")
                        print(f"  .pt:  {pt_top['class']}")
                        print(f"  ONNX: {onnx_top['class']}")
                        mismatches += 1
        
        print()
        
        if mismatches == 0:
            print("✅ ONNX and .pt models produce IDENTICAL results!")
            print()
            print("CONCLUSION:")
            print("  → Problem occurs during ONNX → TFLite conversion")
            print("  → Need to fix TFLite conversion process")
        else:
            print(f"❌ Found {mismatches} mismatches between ONNX and .pt!")
            print()
            print("CONCLUSION:")
            print("  → Problem occurs during .pt → ONNX export")
            print("  → Class mapping gets corrupted in ONNX")
    
    else:
        print("⚠️  .pt results not found, skipping comparison")
    
    print("=" * 80)


if __name__ == "__main__":
    main()