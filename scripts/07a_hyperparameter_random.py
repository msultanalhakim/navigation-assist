"""
Random Search Hyperparameter YOLOv11
Tujuan:
Melakukan eksplorasi acak kombinasi hyperparameter utama (lr, momentum, batch, dll)
untuk menemukan area performa terbaik sebelum grid search.
"""

import os
import random
import csv
import logging
from datetime import datetime
from ultralytics import YOLO
import multiprocessing


def setup_logger(log_dir: str, timestamp: str):
    """Setup logging ke file dan terminal."""
    log_path = os.path.join(log_dir, f"random_search_{timestamp}.log")

    # Konfigurasi dasar logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()  # tampil di terminal juga
        ]
    )

    logging.info(f"Logger initialized. Log file: {log_path}")
    return log_path


def run_random_search():
    # === Konfigurasi Dasar ===
    dataset_yaml = os.path.join(os.getcwd(), "dataset.yaml")
    model_name = os.path.join("models", "yolo11n.pt")
    project_name = "navigation-assistance"
    experiment_name = "random_tuning"

    # === Rentang Parameter ===
    LR_RANGE = (0.001, 0.01)
    MOMENTUM_RANGE = (0.85, 0.95)
    WD_RANGE = (0.0001, 0.001)
    BATCH_OPTIONS = [8, 16]
    OPTIMIZER_OPTIONS = ["SGD", "AdamW"]

    # === Jumlah Eksperimen ===
    N_EXPERIMENTS = 10
    EPOCHS = 30
    IMGSZ = 640

    # === Persiapan Folder & File ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_dir = os.path.join(project_name, experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    log_path = setup_logger(log_dir, timestamp)
    csv_path = os.path.join(log_dir, f"results_random_{timestamp}.csv")

    # === Header CSV ===
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp", "lr0", "momentum", "weight_decay", "batch", "optimizer",
            "map50", "map95", "precision", "recall"
        ])

    logging.info(f"Mulai Random Search ({N_EXPERIMENTS} eksperimen)...")

    # === Loop Eksperimen ===
    for i in range(N_EXPERIMENTS):
        lr0 = round(random.uniform(*LR_RANGE), 5)
        momentum = round(random.uniform(*MOMENTUM_RANGE), 4)
        weight_decay = round(random.uniform(*WD_RANGE), 6)
        batch = random.choice(BATCH_OPTIONS)
        optimizer = random.choice(OPTIMIZER_OPTIONS)

        logging.info(f"[Exp {i+1}/{N_EXPERIMENTS}] lr0={lr0}, momentum={momentum}, wd={weight_decay}, batch={batch}, opt={optimizer}")

        try:
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
                name=f"{experiment_name}_exp{i+1}"
            )

            metrics = model.val()
            m50 = round(metrics.box.map50, 4)
            m95 = round(metrics.box.map, 4)
            mp = round(metrics.box.mp, 4)
            mr = round(metrics.box.mr, 4)

            logging.info(f"[Exp {i+1}] ✅ Done | mAP50={m50}, mAP95={m95}, P={mp}, R={mr}")

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([i + 1, lr0, momentum, weight_decay, batch, optimizer, m50, m95, mp, mr])

        except Exception as e:
            logging.error(f"[Exp {i+1}] ❌ Gagal: {str(e)}")

    logging.info(f"Random Search selesai! Hasil CSV: {csv_path}")
    logging.info(f"Log tersimpan di: {log_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # ✅ Wajib untuk Windows multiprocessing
    run_random_search()
