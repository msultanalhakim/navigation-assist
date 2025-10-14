# Training Model YOLOv11
import os
from ultralytics import YOLO
from datetime import datetime

# === Konfigurasi Dataset & Model ===
dataset_yaml = os.path.join("dataset", "processed", "dataset.yaml")
model_name = os.path.join("models", "yolo11n.pt")

# === Parameter Training ===
epochs = 80
imgsz = 640
batch_size = 8
project_name = "navigation-assistance"  # nama folder utama hasil training
experiment_name = "v1"                  # nama eksperimen (versi model)

# === Inisialisasi Model ===
model = YOLO(model_name)

# === Proses Training ===
results = model.train(
    data=dataset_yaml,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch_size,
    device=0,          # GPU 0 (-1 untuk CPU)
    project=project_name,
    name=experiment_name,
    workers=2,
    patience=10,       # early stopping
    cos_lr=True,       # cosine learning rate
    amp=True,          # mixed precision
    seed=42,
    freeze=10
)

# === Evaluasi & Ringkasan ===
metrics = model.val()

# Format hasil metrik
summary = (
    "\nTraining selesai!\n"
    f"mAP50: {metrics.box.map50:.3f}\n"
    f"mAP50-95: {metrics.box.map:.3f}\n"
    f"Precision: {metrics.box.mp:.3f}\n"
    f"Recall: {metrics.box.mr:.3f}\n"
    f"Model terbaik tersimpan di: {results.save_dir}\n"
)
print(summary)

# === Simpan hasil ke file log ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(project_name, experiment_name)
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"training_log_{timestamp}.txt")

with open(log_path, "w", encoding="utf-8") as f:
    f.write("=== YOLOv11 Training Log ===\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Dataset: {dataset_yaml}\n")
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Image Size: {imgsz}\n")
    f.write(f"Batch Size: {batch_size}\n\n")
    f.write(summary)

print(f"Log hasil training tersimpan di: {log_path}")
