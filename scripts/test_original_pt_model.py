#!/usr/bin/env python3
"""
Test ORIGINAL .pt model (before TFLite conversion)
This will tell us if the problem is in training or conversion
"""

from pathlib import Path
from ultralytics import YOLO
import json

# ============================================================================
# CONFIG
# ============================================================================

# Path to your trained model
MODEL_PT = "models/14-10-2025/best.pt"  # Adjust path!
# Same images you tested with TFLite
SELECTED_IMAGES = [
    "10-129.jpg",                      # Expected: chair
    "65_dup1.jpg",                     # Expected: chair
    "cadeira-de-escritorio-azul_mp4-56.jpg",  # Expected: chair
    "door-7.jpg",                      # Expected: door
    "door-76.jpg",                     # Expected: door
    "WhatsApp-Image-2025-09-29-at-17_28_23-1-_jpeg.rf.11e35fb28317800c5b0a3bc19bf6fbe1.jpg",  # Expected: table
    "WhatsApp-Image-2025-10-10-at-19_30_24-12-_jpeg.rf.2b790aee85d9c76777af8098d4de3434.jpg",  # Expected: ?
    "WhatsApp-Image-2025-09-29-at-17_28_23-1-_jpeg.rf.11e35fb28317800c5b0a3bc19bf6fbe1.jpg",  # Expected: stair
]

DATASET_PATH = Path("dataset") / "processed" / "test" / "images"
CLASS_NAMES = ["person", "chair", "table", "door", "stair"]
CONFIDENCE_THRESHOLD = 0.5

# ============================================================================
# FUNCTIONS
# ============================================================================

def find_image(filename: str) -> Path:
    """Find image by filename"""
    img_path = DATASET_PATH / filename
    if img_path.exists():
        return img_path
    
    for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
        img_path = DATASET_PATH / (Path(filename).stem + ext)
        if img_path.exists():
            return img_path
    
    return None


def test_with_pt_model():
    """Test using original .pt model"""
    
    print("=" * 80)
    print("🧪 TESTING ORIGINAL .pt MODEL (Before TFLite Conversion)")
    print("=" * 80)
    print()
    
    # Check model exists
    if not Path(MODEL_PT).exists():
        print(f"❌ ERROR: Model not found: {MODEL_PT}")
        print("\nPlease update MODEL_PT path in the script to point to your trained model.")
        print("Common locations:")
        print("  - navigation-assistance/v1/weights/best.pt")
        print("  - runs/detect/train/weights/best.pt")
        return
    
    print(f"📦 Loading model: {MODEL_PT}")
    model = YOLO(MODEL_PT)
    print(f"✓ Model loaded")
    print(f"  Classes: {model.names}")
    print()
    
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
    print("RUNNING INFERENCE WITH .pt MODEL")
    print("=" * 80)
    print()
    
    pt_results = []
    class_detections = {cls: 0 for cls in CLASS_NAMES}
    
    for idx, (filename, img_path) in enumerate(valid_images, 1):
        print(f"[{idx}/{len(valid_images)}] {filename}")
        
        # Run prediction
        results = model.predict(
            source=str(img_path),
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )
        
        # Parse results
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                
                if cls_id < len(CLASS_NAMES):
                    cls_name = CLASS_NAMES[cls_id]
                    
                    detections.append({
                        'class': cls_name,
                        'class_id': cls_id,
                        'confidence': conf
                    })
                    
                    class_detections[cls_name] += 1
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Store result
        pt_results.append({
            'filename': filename,
            'detections': detections[:10]  # Top 10
        })
        
        # Print detections
        if detections:
            for i, det in enumerate(detections[:10]):
                marker = "→" if i == 0 else " "
                print(f"  {marker} {det['class']:10s} conf={det['confidence']:.4f}")
        else:
            print(f"  ⚠️  No detections above {CONFIDENCE_THRESHOLD}")
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY - .pt MODEL RESULTS")
    print("=" * 80)
    print()
    
    print("Detections by class:")
    for cls_name in CLASS_NAMES:
        count = class_detections[cls_name]
        print(f"  {cls_name:10s}: {count:3d}")
    
    print()
    
    # Save results
    with open('pt_model_test_results.json', 'w') as f:
        json.dump({
            'model': MODEL_PT,
            'results': pt_results,
            'summary': class_detections
        }, f, indent=2)
    
    print("📄 Results saved to: pt_model_test_results.json")
    print("=" * 80)
    
    return pt_results, class_detections


def compare_with_tflite():
    """Compare .pt results with TFLite results"""
    
    print()
    print("=" * 80)
    print("📊 COMPARISON: .pt vs TFLite")
    print("=" * 80)
    print()
    
    # Try to load TFLite results
    tflite_results_file = "selective_test_results.json"
    
    if not Path(tflite_results_file).exists():
        print("⚠️  TFLite results not found (selective_test_results.json)")
        print("   Run the TFLite test first to enable comparison")
        return
    
    with open(tflite_results_file, 'r') as f:
        tflite_data = json.load(f)
    
    with open('pt_model_test_results.json', 'r') as f:
        pt_data = json.load(f)
    
    print("Comparing detections for each image:\n")
    
    # Create lookup for easier comparison
    tflite_lookup = {r['image'] if 'image' in r else r['filename']: r for r in tflite_data['results']}
    pt_lookup = {r['filename']: r for r in pt_data['results']}
    
    mismatches = []
    
    for filename in pt_lookup.keys():
        print(f"{filename}:")
        
        # Get top detection from each
        pt_top = pt_lookup[filename]['detections'][0] if pt_lookup[filename]['detections'] else None
        
        # Find in TFLite results
        tflite_result = None
        for key in tflite_lookup.keys():
            if filename in key or key in filename:
                tflite_result = tflite_lookup[key]
                break
        
        tflite_top = tflite_result['detections'][0] if tflite_result and tflite_result['detections'] else None
        
        if pt_top and tflite_top:
            pt_class = pt_top['class']
            tflite_class = tflite_top['class']
            
            if pt_class == tflite_class:
                print(f"  ✓ MATCH: Both detected '{pt_class}'")
            else:
                print(f"  ❌ MISMATCH:")
                print(f"     .pt model:    {pt_class} (conf={pt_top['confidence']:.4f})")
                print(f"     TFLite model: {tflite_class} (conf={tflite_top['confidence']:.4f})")
                mismatches.append(filename)
        elif pt_top and not tflite_top:
            print(f"  ⚠️  .pt detected '{pt_top['class']}', TFLite detected nothing")
        elif not pt_top and tflite_top:
            print(f"  ⚠️  .pt detected nothing, TFLite detected '{tflite_top['class']}'")
        else:
            print(f"  ⚠️  Both detected nothing")
        
        print()
    
    # Summary
    print("=" * 80)
    print("🎯 DIAGNOSIS")
    print("=" * 80)
    print()
    
    if not mismatches:
        print("✅ RESULT: .pt and TFLite models produce IDENTICAL results!")
        print()
        print("CONCLUSION:")
        print("  → TFLite conversion is CORRECT")
        print("  → Testing code is CORRECT")
        print("  → The problem is in TRAINING (freeze=10)")
        print()
        print("RECOMMENDATION:")
        print("  → Retrain with freeze=0")
        print("  → Use the fixed training script")
    else:
        print(f"❌ RESULT: Found {len(mismatches)} MISMATCHES between .pt and TFLite!")
        print()
        print("Files with different predictions:")
        for f in mismatches:
            print(f"  - {f}")
        print()
        print("CONCLUSION:")
        print("  → TFLite conversion has ISSUES")
        print("  → Need to fix conversion process")
        print()
        print("RECOMMENDATION:")
        print("  → Check TFLite conversion script")
        print("  → Verify quantization settings")
        print("  → Try float32 instead of int8")
    
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        # Test .pt model
        pt_results, class_counts = test_with_pt_model()
        
        if pt_results:
            # Compare with TFLite
            compare_with_tflite()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()