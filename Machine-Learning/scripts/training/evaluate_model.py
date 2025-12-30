"""
YOLOv11 Model Evaluation Script with Config Integration
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path

from scripts.utils import logger, config


def main():
    """Main evaluation function"""
    
    # Get paths from config or use defaults
    project = config.get('project', 'runs/train')
    experiment = config.get('experiment', 'exp_latest')
    
    # Construct paths
    base_dir = os.path.join(project, experiment)
    best_model_path = os.path.join(base_dir, "weights", "best.pt")
    results_csv = os.path.join(base_dir, "results.csv")
    results_png = os.path.join(base_dir, "results.png")
    
    # Check if model exists
    if not os.path.exists(best_model_path):
        logger.error(f"Model not found: {best_model_path}")
        logger.info("Please specify correct project and experiment in config.yaml")
        return
    
    logger.info(f"Loading best model from: {best_model_path}")
    model = YOLO(best_model_path)
    
    # Evaluate model on validation set
    logger.info("Evaluating model on validation set...")
    metrics = model.val()
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"mAP50: {metrics.box.map50:.4f}")
    logger.info(f"mAP50-95: {metrics.box.map:.4f}")
    logger.info(f"Precision: {metrics.box.mp:.4f}")
    logger.info(f"Recall: {metrics.box.mr:.4f}")
    logger.info("=" * 80)
    
    # Read training results CSV for additional analysis
    if os.path.exists(results_csv):
        logger.info("\nTraining Statistics (last 5 epochs):")
        df = pd.read_csv(results_csv)
        
        # Select relevant columns
        relevant_cols = []
        for col in ["epoch", "train/box_loss", "val/box_loss", 
                    "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)"]:
            if col in df.columns:
                relevant_cols.append(col)
        
        if relevant_cols:
            logger.info("\n" + str(df[relevant_cols].tail(5)))
    else:
        logger.warning("results.csv not found")
    
    # Display training results plot
    if os.path.exists(results_png):
        logger.info(f"\nTraining plot available at: {results_png}")
        try:
            img = plt.imread(results_png)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis("off")
            plt.title("YOLOv11 Training Results", fontsize=14)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"Could not display plot: {e}")
    else:
        logger.warning("results.png not found")
    
    # Save evaluation log
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(base_dir, f"evaluation_log_{timestamp}.txt")
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("YOLOv11 EVALUATION LOG\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("MODEL INFORMATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model Path: {best_model_path}\n")
        f.write(f"Project: {project}\n")
        f.write(f"Experiment: {experiment}\n")
        f.write(f"Classes: {config.class_names}\n")
        f.write(f"Number of Classes: {config.num_classes}\n\n")
        
        f.write("EVALUATION RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"mAP50: {metrics.box.map50:.4f}\n")
        f.write(f"mAP50-95: {metrics.box.map:.4f}\n")
        f.write(f"Precision: {metrics.box.mp:.4f}\n")
        f.write(f"Recall: {metrics.box.mr:.4f}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 80 + "\n")
        if hasattr(metrics.box, 'ap_class_index'):
            for i, class_idx in enumerate(metrics.box.ap_class_index):
                class_name = config.class_names[int(class_idx)]
                f.write(f"{class_name}:\n")
                f.write(f"  mAP50: {metrics.box.ap50[i]:.4f}\n")
                f.write(f"  mAP50-95: {metrics.box.ap[i]:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Evaluation completed at: {timestamp}\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"\nEvaluation log saved to: {log_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()