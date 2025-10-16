"""
Script untuk konversi YOLOv11 model (.pt) ke TFLite format
Untuk deployment ke Android
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def convert_to_tflite(
    model_path: str,
    output_dir: str = ".",
    imgsz: int = 640,
    int8: bool = False
):
    """
    Konversi YOLO model ke TFLite
    
    Args:
        model_path: Path ke .pt model
        output_dir: Directory output
        imgsz: Input size (default 640)
        int8: Use INT8 quantization (smaller but less accurate)
    """
    
    print("=" * 80)
    print("🔄 YOLO TO TFLITE CONVERTER")
    print("=" * 80)
    
    # Validate model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model tidak ditemukan di {model_path}")
        sys.exit(1)
    
    print(f"📂 Input Model: {model_path}")
    print(f"📁 Output Dir: {output_dir}")
    print(f"📐 Image Size: {imgsz}x{imgsz}")
    print(f"🔢 Quantization: {'INT8' if int8 else 'FP32'}")
    print()
    
    try:
        # Load YOLO model
        print("⏳ Loading model...")
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
        print()
        
        # Model info
        print("📊 Model Info:")
        print(f"  • Classes: {len(model.names)}")
        print(f"  • Names: {list(model.names.values())}")
        print()
        
        # Export to TFLite
        print("🚀 Exporting to TFLite...")
        print("   This may take a few minutes...")
        
        export_path = model.export(
            format="tflite",
            imgsz=imgsz,
            int8=int8,
            data=None  # No calibration dataset needed for FP32
        )
        
        print()
        print("=" * 80)
        print("✅ CONVERSION SUCCESSFUL!")
        print("=" * 80)
        print(f"📦 Output: {export_path}")
        
        # File info
        file_size = os.path.getsize(export_path) / (1024 * 1024)  # MB
        print(f"📊 File Size: {file_size:.2f} MB")
        print()
        
        # Instructions
        print("📝 NEXT STEPS:")
        print("-" * 80)
        print(f"1. Copy file ke Android project:")
        print(f"   cp {export_path} <YourProject>/app/src/main/assets/best.tflite")
        print()
        print("2. Verify model di YoloDetector.kt:")
        print(f"   modelPath = \"best.tflite\"")
        print()
        print("3. Update class labels di YoloDetector.kt:")
        print(f"   labels = listOf{tuple(model.names.values())}")
        print()
        print("4. Build & Run aplikasi Android")
        print("=" * 80)
        
        return export_path
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ CONVERSION FAILED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print()
        print("💡 Troubleshooting:")
        print("  • Pastikan ultralytics versi terbaru: pip install -U ultralytics")
        print("  • Check model format: file harus .pt")
        print("  • Coba tanpa quantization: int8=False")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert YOLO model to TFLite")
    parser.add_argument(
        "--model",
        type=str,
        default="navigation-assistance/v12/weights/best.pt",
        help="Path to .pt model file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 quantization (smaller file, slightly less accurate)"
    )
    
    args = parser.parse_args()
    
    convert_to_tflite(
        model_path=args.model,
        output_dir=args.output,
        imgsz=args.imgsz,
        int8=args.int8
    )