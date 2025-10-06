"""
Evaluate trained YOLOv11 model and return evaluation metrics
"""
from pathlib import Path
import logging
from ultralytics import YOLO  # pylint: disable=E0611

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def evaluate_model() -> dict:
    """
    Evaluate YOLOv11 model on validation set and return metrics as a dictionary
    """
    root = Path(__file__).resolve().parent

    # Path model dan data
    model_path = root / "models" / "optimized" / "final_model" / "weights" / "best.pt"
    dataset_yaml = root / "dataset" / "data.yaml"

    logging.info("Loading trained model from: %s", model_path)
    model = YOLO(str(model_path))

    logging.info("Evaluating model performance...")
    results = model.val(
        data=str(dataset_yaml),
        conf=0.001,      # low threshold untuk recall tinggi
        iou=0.7,         # IoU untuk NMS saat evaluasi
        plots=True,      # simpan confusion matrix & PR curve
        save_json=True   # simpan hasil dalam format COCO JSON
    )

    # Ambil metrik penting
    metrics = {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "precision": results.box.p.tolist(),  # list per kelas
        "recall": results.box.r.tolist()      # list per kelas
    }

    logging.info("Evaluation complete: %s", metrics)
    return metrics


if __name__ == "__main__":
    eval_results = evaluate_model()
    print("\nFinal Evaluation Metrics:")
    for key, value in eval_results.items():
        print(f"{key}: {value}")
