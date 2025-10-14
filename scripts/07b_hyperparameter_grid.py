"""
📗 07b_hyperparameter_grid.py — Grid Search Hyperparameter YOLOv11
Tujuan:
Melakukan eksplorasi sistematis kombinasi hyperparameter dari range yang dipersempit
berdasarkan hasil terbaik Random Search.
"""

import os
import csv
from itertools import product
from datetime import datetime
from ultralytics import YOLO

# === Konfigurasi Dasar ===
dataset_yaml = os.path.join("dataset", "processed", "dataset.yaml")
model_name = os.path.join("models", "yolo11n.pt")
project_name = "navigation-assistance"
experiment_name = "tuning_grid"

# === Nilai Grid (berdasarkan hasil random search terbaik) ===
LR_VALUES = [0.003, 0.005, 0.007]
MOMENTUM_VALUES = [0.88, 0.9, 0.92]
WD_VALUES = [0.0003, 0.0005]
BATCH_VALUES = [8, 16]
OPTIMIZER_VALUES = ["SGD", "AdamW"]

# === Parameter Umum ===
EPOCHS = 30
IMGSZ = 640

# === Persiapan Folder & CSV ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
log_dir = os.path.join(project_name, experiment_name)
os.makedirs(log_dir, exist_ok=True)
csv_path = os.path.join(log_dir, f"results_grid_{timestamp}.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["lr0", "momentum", "weight_decay", "batch", "optimizer", "map50", "map95", "precision", "recall"])

# === Loop Grid Kombinasi ===
combinations = list(product(LR_VALUES, MOMENTUM_VALUES, WD_VALUES, BATCH_VALUES, OPTIMIZER_VALUES))
print(f"Total kombinasi grid: {len(combinations)}\n")

for i, (lr0, momentum, weight_decay, batch, optimizer) in enumerate(combinations, 1):
    print(f"🚀 [Grid {i}/{len(combinations)}] lr0={lr0}, momentum={momentum}, wd={weight_decay}, batch={batch}, opt={optimizer}")

    model = YOLO(model_name)
    results = model.train(
        data=dataset_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=batch,
        lr0=lr0,
        momentum=momentum,
        weight_decay=weight_decay,
        optimizer=optimizer,
        device=0,
        workers=2,
        patience=10,
        project=project_name,
        name=f"{experiment_name}_grid{i}"
    )

    metrics = model.val()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            lr0,
            momentum,
            weight_decay,
            batch,
            optimizer,
            round(metrics.box.map50, 4),
            round(metrics.box.map, 4),
            round(metrics.box.mp, 4),
            round(metrics.box.mr, 4)
        ])

print(f"\nGrid Search selesai! Hasil tersimpan di: {csv_path}")
