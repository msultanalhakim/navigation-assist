"""
Mengonversi model hasil training (best.pt) menjadi format TensorFlow Lite (.tflite)
agar dapat di-deploy ke aplikasi Android.
"""

import os
from ultralytics import YOLO
from datetime import datetime

# === Konfigurasi Path Model ===
trained_model = os.path.join("navigation-assistance", "v1", "weights", "best.pt")  # hasil training
output_dir = os.path.join("models", "exported")  # folder penyimpanan hasil ekspor
os.makedirs(output_dir, exist_ok=True)

# === Load Model ===
print(f"Memuat model: {trained_model}")
model = YOLO(trained_model)

# === Ekspor ke TensorFlow Lite ===
print("⚙️ Mengonversi model ke TensorFlow Lite...")
results = model.export(
    format="tflite",     # format export
    imgsz=640,           # resolusi input (harus sama seperti saat training)
    dynamic=False,       # True → untuk input ukuran dinamis
    optimize=True,       # optimasi kuantisasi ukuran model
    half=False           # False → gunakan FP32 (True untuk FP16 jika didukung)
)

# === Informasi hasil ekspor ===
tflite_path = os.path.join(output_dir, "best.tflite")
if os.path.exists(tflite_path):
    print(f"\nKonversi berhasil! Model tersimpan di:\n📂 {tflite_path}")
else:
    print("\nKonversi selesai, tetapi file best.tflite tidak ditemukan otomatis.")
    print(f"Periksa folder hasil: {results}")

# === Logging hasil ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(output_dir, f"export_log_{timestamp}.txt")

with open(log_path, "w", encoding="utf-8") as f:
    f.write("=== YOLOv11 → TensorFlow Lite Export Log ===\n")
    f.write(f"Model Source: {trained_model}\n")
    f.write(f"TFLite Output: {tflite_path}\n")
    f.write(f"Image Size: 640x640\n")
    f.write(f"Optimize: True\n")
    f.write(f"Dynamic Input: False\n")
    f.write(f"Timestamp: {timestamp}\n")

print(f"Log ekspor tersimpan di: {log_path}")
