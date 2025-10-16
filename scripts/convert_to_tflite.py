#!/usr/bin/env python3
"""
Fixed TFLite Conversion - Preserve Class Mapping
PROBLEM: Original conversion causes class ID shifting
SOLUTION: Export with proper metadata and verify class order
"""

from ultralytics import YOLO
import tensorflow as tf
import json
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

MODEL_PT = "models/14-10-2025/best.pt"  # Your trained model
OUTPUT_DIR = "models/tflite_fixed"
CLASS_NAMES = ["person", "chair", "table", "door", "stair"]

# ============================================================================
# CONVERSION
# ============================================================================

def convert_to_tflite():
    """
    Convert YOLOv11 .pt to TFLite with proper class mapping
    """
    
    print("=" * 80)
    print("🔧 FIXED TFLite CONVERSION")
    print("=" * 80)
    print()
    
    # Check model exists
    if not Path(MODEL_PT).exists():
        print(f"❌ ERROR: Model not found: {MODEL_PT}")
        return
    
    print(f"📦 Loading model: {MODEL_PT}")
    model = YOLO(MODEL_PT)
    
    # Verify class names
    print(f"✓ Model loaded")
    print(f"  Classes: {model.names}")
    print()
    
    # Verify class order
    print("🔍 Verifying class order:")
    for i, name in model.names.items():
        expected = CLASS_NAMES[i]
        status = "✓" if name == expected else "❌"
        print(f"  {status} Class {i}: {name} (expected: {expected})")
    
    print()
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # METHOD 1: Direct TFLite Export (Ultralytics)
    # ========================================================================
    
    print("=" * 80)
    print("METHOD 1: Ultralytics Export (Recommended)")
    print("=" * 80)
    print()
    
    try:
        print("🔄 Converting to TFLite (float32)...")
        
        # Export to TFLite
        tflite_path = model.export(
            format='tflite',
            imgsz=640,
            int8=False,  # Use float32 for accuracy
            half=False,
        )
        
        print(f"✅ Conversion successful!")
        print(f"📁 TFLite model: {tflite_path}")
        
        # Copy to output directory with better name
        import shutil
        output_path = Path(OUTPUT_DIR) / "best_float32.tflite"
        shutil.copy(tflite_path, output_path)
        print(f"📁 Copied to: {output_path}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return
    
    print()
    
    # ========================================================================
    # VERIFY CONVERSION
    # ========================================================================
    
    print("=" * 80)
    print("🔍 VERIFYING TFLITE MODEL")
    print("=" * 80)
    print()
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(output_path))
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("Input details:")
        for detail in input_details:
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Type: {detail['dtype']}")
        
        print()
        print("Output details:")
        for detail in output_details:
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Type: {detail['dtype']}")
        
        print()
        
        # Check output shape
        output_shape = output_details[0]['shape']
        print(f"Output shape: {output_shape}")
        
        # For YOLOv11: [batch, 9 (4 bbox + 5 classes), num_predictions]
        if len(output_shape) == 3:
            num_classes = output_shape[1] - 4  # Subtract bbox coords
            print(f"Number of classes in output: {num_classes}")
            
            if num_classes != len(CLASS_NAMES):
                print(f"⚠️  WARNING: Expected {len(CLASS_NAMES)} classes, got {num_classes}")
            else:
                print(f"✓ Class count matches ({num_classes} classes)")
        
    except Exception as e:
        print(f"⚠️  Verification error: {e}")
    
    print()
    
    # ========================================================================
    # SAVE METADATA
    # ========================================================================
    
    print("=" * 80)
    print("💾 SAVING METADATA")
    print("=" * 80)
    print()
    
    metadata = {
        'model_source': MODEL_PT,
        'class_names': CLASS_NAMES,
        'class_mapping': {i: name for i, name in enumerate(CLASS_NAMES)},
        'input_size': 640,
        'format': 'tflite',
        'quantization': 'float32',
        'notes': [
            'Class order is CRITICAL!',
            'Output format: [batch, 9, num_predictions]',
            'First 4 values: [x, y, w, h]',
            'Next 5 values: class probabilities [person, chair, table, door, stair]'
        ]
    }
    
    metadata_path = Path(OUTPUT_DIR) / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {metadata_path}")
    print()
    
    # ========================================================================
    # TESTING SNIPPET
    # ========================================================================
    
    print("=" * 80)
    print("📝 TESTING CODE SNIPPET")
    print("=" * 80)
    print()
    print("Use this code to load the model correctly:")
    print()
    print("```python")
    print("import tensorflow as tf")
    print("import json")
    print()
    print("# Load model")
    print(f"interpreter = tf.lite.Interpreter('{output_path}')")
    print("interpreter.allocate_tensors()")
    print()
    print("# Load class names from metadata")
    print(f"with open('{metadata_path}') as f:")
    print("    metadata = json.load(f)")
    print("    CLASS_NAMES = metadata['class_names']")
    print()
    print("# IMPORTANT: Use CLASS_NAMES for mapping predictions!")
    print("# class_id 0 → CLASS_NAMES[0] = 'person'")
    print("# class_id 1 → CLASS_NAMES[1] = 'chair'")
    print("# etc.")
    print("```")
    print()
    
    print("=" * 80)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 80)
    print()
    print("📁 Output files:")
    print(f"  - {output_path}")
    print(f"  - {metadata_path}")
    print()
    print("🧪 Next steps:")
    print("  1. Test with: python test_tflite_fixed.py")
    print("  2. Verify class predictions are now correct")
    print("  3. If still wrong, check your testing code's class mapping")
    print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    convert_to_tflite()