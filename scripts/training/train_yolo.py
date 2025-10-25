"""
YOLOv11 Training Script with Config Integration
"""

from ultralytics import YOLO
import os
import time
import json
from datetime import datetime
from pathlib import Path
import torch
import gc

from scripts.utils import logger, config


class TrainingConfig:
    """Training configuration from config.yaml and defaults"""
    
    def __init__(self):
        # Load from config.yaml
        self.dataset_yaml = config.get('dataset_yaml', 'config.yaml')
        self.model_path = config.get('model', 'yolo11n.pt')
        
        # Training parameters
        training_params = config.get('training', {})
        self.epochs = training_params.get('epochs', 100)
        self.batch = training_params.get('batch', 16)
        self.imgsz = training_params.get('imgsz', 640)
        self.patience = training_params.get('patience', 50)
        self.freeze = training_params.get('freeze', 10)
        self.lr0 = training_params.get('lr0', 0.01)
        self.momentum = training_params.get('momentum', 0.937)
        self.weight_decay = training_params.get('weight_decay', 0.0005)
        self.optimizer = training_params.get('optimizer', 'SGD')
        self.cos_lr = training_params.get('cos_lr', True)
        
        # System
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.workers = training_params.get('workers', 2)
        
        # Output
        self.project = config.get('project', 'runs/train')
        self.experiment = config.get('experiment', f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def cleanup_gpu():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def format_time(seconds):
    """Format seconds to human readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def main():
    """Main training function"""
    
    # Load configuration
    cfg = TrainingConfig()
    
    # Setup directories
    run_dir = os.path.join(cfg.project, cfg.experiment)
    os.makedirs(run_dir, exist_ok=True)
    
    config_json = os.path.join(run_dir, "config.json")
    resume_flag = os.path.join(run_dir, "resume.txt")
    
    # Save config for resume
    if not os.path.exists(config_json):
        with open(config_json, "w") as f:
            json.dump({
                "dataset": cfg.dataset_yaml,
                "model": cfg.model_path,
                "epochs": cfg.epochs,
                "batch": cfg.batch,
                "imgsz": cfg.imgsz,
                "patience": cfg.patience,
                "freeze": cfg.freeze,
                "lr0": cfg.lr0,
                "momentum": cfg.momentum,
                "weight_decay": cfg.weight_decay,
                "optimizer": cfg.optimizer,
                "cos_lr": cfg.cos_lr,
                "started_at": cfg.experiment
            }, f, indent=2)
    
    # Check resume
    resume_training = os.path.exists(resume_flag)
    last_checkpoint = os.path.join(run_dir, "weights", "last.pt")
    
    if resume_training and not os.path.exists(last_checkpoint):
        logger.warning("Checkpoint not found, starting from scratch")
        resume_training = False
    
    # Training info
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info("YOLOv11 Training Configuration")
    logger.info(f"Output: {run_dir}")
    logger.info(f"Device: {gpu_name}")
    logger.info(f"Epochs: {cfg.epochs} | Batch: {cfg.batch} | Patience: {cfg.patience}")
    logger.info(f"Resume: {'Yes' if resume_training else 'No'}")
    
    # Training
    training_start = time.time()
    status = "FAILED"
    map50 = map95 = mp = mr = 0.0
    
    try:
        # Mark training in progress
        with open(resume_flag, "w") as f:
            f.write(f"Training started: {cfg.experiment}\n")
        
        # Load model
        if resume_training:
            logger.info(f"Loading checkpoint: {last_checkpoint}")
            model = YOLO(last_checkpoint)
        else:
            logger.info(f"Loading model: {cfg.model_path}")
            model = YOLO(cfg.model_path)
        
        logger.info("Starting training...")
        
        # Train
        results = model.train(
            data=cfg.dataset_yaml,
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            lr0=cfg.lr0,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            optimizer=cfg.optimizer,
            cos_lr=cfg.cos_lr,
            device=cfg.device,
            workers=cfg.workers,
            patience=cfg.patience,
            freeze=cfg.freeze,
            project=cfg.project,
            name=cfg.experiment,
            exist_ok=True,
            pretrained=not resume_training,
            resume=resume_training,
            verbose=True
        )
        
        training_duration = time.time() - training_start
        
        # Final validation
        logger.info("Running final validation...")
        val_results = model.val()
        
        map50 = float(val_results.box.map50)
        map95 = float(val_results.box.map)
        mp = float(val_results.box.mp)
        mr = float(val_results.box.mr)
        status = "SUCCESS"
        
        # Remove resume flag
        if os.path.exists(resume_flag):
            os.remove(resume_flag)
        
        # Save summary
        summary_path = os.path.join(run_dir, "training_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("YOLOv11 TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Model: {cfg.model_path}\n")
            f.write(f"Dataset: {cfg.dataset_yaml}\n")
            f.write(f"Epochs: {cfg.epochs}\n")
            f.write(f"Image Size: {cfg.imgsz}\n")
            f.write(f"Batch Size: {cfg.batch}\n")
            f.write(f"Learning Rate: {cfg.lr0}\n")
            f.write(f"Momentum: {cfg.momentum}\n")
            f.write(f"Weight Decay: {cfg.weight_decay}\n")
            f.write(f"Optimizer: {cfg.optimizer}\n")
            f.write(f"Cosine LR: {cfg.cos_lr}\n")
            f.write(f"Device: {cfg.device}\n")
            f.write(f"Patience: {cfg.patience}\n")
            f.write(f"Freeze: {cfg.freeze} layers\n\n")
            
            f.write("RESULTS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Training Duration: {format_time(training_duration)}\n")
            f.write(f"mAP50: {map50:.4f}\n")
            f.write(f"mAP50-95: {map95:.4f}\n")
            f.write(f"Precision: {mp:.4f}\n")
            f.write(f"Recall: {mr:.4f}\n")
            f.write(f"Status: {status}\n\n")
            
            f.write("SAVED FILES:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Best weights: {Path(results.save_dir) / 'weights' / 'best.pt'}\n")
            f.write(f"Last weights: {Path(results.save_dir) / 'weights' / 'last.pt'}\n")
            f.write("=" * 80 + "\n")
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"Duration: {format_time(training_duration)}")
        logger.info(f"mAP50: {map50:.4f}")
        logger.info(f"mAP50-95: {map95:.4f}")
        logger.info(f"Precision: {mp:.4f}")
        logger.info(f"Recall: {mr:.4f}")
        logger.info(f"Best weights: {Path(results.save_dir) / 'weights' / 'best.pt'}")
        logger.info(f"Summary: {summary_path}")
        logger.info("=" * 80)
        
        cleanup_gpu()
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        logger.info(f"Checkpoint saved at: {run_dir}/weights/last.pt")
        logger.info("Run this script again to resume training")
        cleanup_gpu()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        cleanup_gpu()
        raise


if __name__ == "__main__":
    main()