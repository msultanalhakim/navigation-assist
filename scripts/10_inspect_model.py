"""
Script untuk mengecek Input/Output Shape dari YOLOv11 model (.pt)
"""

from ultralytics import YOLO
import torch

# Path ke model Anda (sesuaikan dengan lokasi hasil training)
model_path = "navigation-assistance/v12/weights/best.pt"

print("=" * 80)
print("🔍 CHECKING YOLO MODEL SHAPES")
print("=" * 80)

# Load model
model = YOLO(model_path)
print(f"\n✅ Model loaded: {model_path}\n")

# ==============================================================================
# METODE 1: Info dari Model
# ==============================================================================
print("📊 METHOD 1: Model Info")
print("-" * 80)
print(model.model)  # Print architecture
print()

# ==============================================================================
# METODE 2: Test dengan Dummy Input
# ==============================================================================
print("📊 METHOD 2: Dummy Input Test")
print("-" * 80)

# Buat dummy input (batch_size=1, channels=3, height=640, width=640)
dummy_input = torch.randn(1, 3, 640, 640)
print(f"Input Shape: {dummy_input.shape}")
print(f"  └─ Format: [batch_size, channels, height, width]")
print(f"  └─ Values: [{dummy_input.shape[0]}, {dummy_input.shape[1]}, {dummy_input.shape[2]}, {dummy_input.shape[3]}]")

# Forward pass
model.model.eval()
with torch.no_grad():
    output = model.model(dummy_input)

print(f"\nOutput Type: {type(output)}")

# YOLO output biasanya berupa tuple atau list
if isinstance(output, (tuple, list)):
    print(f"Number of outputs: {len(output)}")
    for i, out in enumerate(output):
        if isinstance(out, torch.Tensor):
            print(f"\nOutput[{i}] Shape: {out.shape}")
            print(f"  └─ Total elements: {out.numel()}")
else:
    print(f"Output Shape: {output.shape}")

print()

# ==============================================================================
# METODE 3: Model Info Detail
# ==============================================================================
print("📊 METHOD 3: Detailed Model Info")
print("-" * 80)

# Info umum
info = model.info(detailed=False, verbose=True)

print()

# ==============================================================================
# METODE 4: Prediction Test (Real Use Case)
# ==============================================================================
print("📊 METHOD 4: Real Prediction Test")
print("-" * 80)

# Test dengan image dummy
import numpy as np
from PIL import Image

# Buat dummy image
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
dummy_image_pil = Image.fromarray(dummy_image)

# Predict
results = model.predict(dummy_image_pil, verbose=False)

print(f"Input: PIL Image or numpy array (H, W, C)")
print(f"  └─ Example shape: (640, 640, 3)")
print(f"\nOutput: Results object")
print(f"  └─ Type: {type(results[0])}")
print(f"  └─ Boxes: {results[0].boxes.shape if results[0].boxes is not None else 'None'}")
print(f"  └─ Number of detections: {len(results[0].boxes) if results[0].boxes is not None else 0}")

if len(results[0].boxes) > 0:
    print(f"\n📦 Box format: [x1, y1, x2, y2, confidence, class]")
    print(f"  └─ boxes.xyxy: Bounding box coordinates")
    print(f"  └─ boxes.conf: Confidence scores")
    print(f"  └─ boxes.cls: Class indices")

print()

# ==============================================================================
# SUMMARY
# ==============================================================================
print("=" * 80)
print("📝 SUMMARY")
print("=" * 80)
print(f"""
INPUT SHAPE:
  • Tensor format: torch.Tensor([1, 3, 640, 640])
  • Image format: numpy.ndarray (H, W, C) atau PIL.Image
  • Dimensions: 640x640 pixels (default, bisa diubah saat inference)
  • Channels: 3 (RGB)

OUTPUT SHAPE:
  • Detection boxes: [N, 6] dimana N = jumlah deteksi
  • Format per box: [x1, y1, x2, y2, confidence, class_id]
  • Class names: {model.names}

USAGE:
  results = model.predict("image.jpg")
  boxes = results[0].boxes.xyxy   # Koordinat [x1,y1,x2,y2]
  conf = results[0].boxes.conf     # Confidence scores
  cls = results[0].boxes.cls       # Class IDs
""")
print("=" * 80)