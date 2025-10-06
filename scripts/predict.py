"""
Run inference with trained YOLOv11 model on test images
"""
from pathlib import Path
import logging
from ultralytics import YOLO  # pylint: disable=E0611

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def predict_images(conf_threshold=0.5) -> list:
    """
    Predict test images and return list of results
    """
    root = Path(__file__).resolve().parent
    model_path = root / "models" / "optimized" / "final_model" / "weights" / "best.pt"
    source = root / "dataset" / "test" / "images"

    logging.info("Loading model: %s", model_path)
    model = YOLO(str(model_path))

    logging.info("Running inference on: %s", source)
    results = model.predict(
        source=str(source),
        conf=conf_threshold,
        save=True,
        save_conf=True,
        imgsz=640,
        max_det=300
    )

    # Return semua hasil dalam list
    logging.info("Predictions complete. Saved to runs/detect/predict/")
    return results


if __name__ == "__main__":
    output = predict_images()
    print("\nPrediction Finished. Total results:", len(output))
