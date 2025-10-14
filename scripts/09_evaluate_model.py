# Evaluasi Model YOLOv11
import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from datetime import datetime

# === Path model dan dataset ===
project_name = "navigation-assistance"
experiment_name = "v1"
base_dir = os.path.join(project_name, experiment_name)
best_model_path = os.path.join(base_dir, "weights", "best.pt")
results_csv = os.path.join(base_dir, "results.csv")
results_png = os.path.join(base_dir, "results.png")

if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"Model terbaik tidak ditemukan: {best_model_path}")

print(f"📂 Memuat model terbaik dari: {best_model_path}")
model = YOLO(best_model_path)

# === Evaluasi model di validation set ===
print("\nMengevaluasi model pada validation set...")
metrics = model.val()

summary = (
    "\nEvaluasi selesai!\n"
    f"mAP50: {metrics.box.map50:.3f}\n"
    f"mAP50-95: {metrics.box.map:.3f}\n"
    f"Precision: {metrics.box.mp:.3f}\n"
    f"Recall: {metrics.box.mr:.3f}\n"
)
print(summary)

# === Baca results.csv untuk analisis tambahan ===
if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)
    print("Statistik Training (ringkasan 5 epoch terakhir):")
    print(df.tail(5)[["epoch", "train/box_loss", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)"]])
else:
    print("File results.csv tidak ditemukan.")

# === Tampilkan grafik hasil training ===
if os.path.exists(results_png):
    img = plt.imread(results_png)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Grafik Hasil Training YOLOv11", fontsize=11)
    plt.show()
else:
    print("Gambar results.png tidak ditemukan.")

# === Simpan log evaluasi ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(base_dir, f"evaluation_log_{timestamp}.txt")

with open(log_path, "w", encoding="utf-8") as f:
    f.write("=== YOLOv11 Evaluation Log ===\n")
    f.write(f"Model: {best_model_path}\n")
    f.write(f"Project: {project_name}\n")
    f.write(f"Experiment: {experiment_name}\n\n")
    f.write(summary)

print(f"Log hasil evaluasi tersimpan di: {log_path}")
