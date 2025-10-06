"""
Train YOLOv11 fine-tuning from COCO -> Custom 5 Classes
"""

from pathlib import Path
import logging
from ultralytics import YOLO  # pylint: disable=E0611

# Setup logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_model() -> dict:
    """
    Train YOLOv11 model and return training metrics
    """
    root = Path(__file__).resolve().parent.parent
    # Path ke dataset
    dataset_yaml = root / "dataset" / "data.yaml"

    # Folder output model
    out_dir = root / "models" / "optimized"

    # Load pretrained YOLOv11 (COCO)
    pretrained = "yolo11n.pt"  # otomatis download jika belum ada
    logging.info("Loading YOLOv11 pretrained model: %s", pretrained)
    model = YOLO(pretrained)

    logging.info("Starting training...")
    results = model.train(
        data=str(dataset_yaml),
        epochs=100,
        imgsz=640,
        batch=8,             # turunkan ke 4/2 jika OOM
        device=0,            # "cpu" jika tidak ada GPU
        project=str(out_dir),
        name="final_model",
        seed=42,
        workers=0,           # stabil untuk Windows
        patience=20,         # early stopping
        optimizer="auto",
        cos_lr=True,
        amp=True,
        save_period=-1,      # hemat disk space
        pretrained=True
    )

    logging.info("Training selesai. Best weights di: %s",
                 out_dir / "final_model" / "weights" / "best.pt")

    # Return hasil dalam dict
    metrics = {
        "epochs_ran": results.epoch,
        "metrics": results.metrics,
        "final_dir": str(out_dir / "final_model"),
    }
    return metrics


if __name__ == "__main__":
    output = train_model()
    print("\nTraining Finished ->")
    print(output)
