"""
YOLOv11 Training - FIXED VERSION
Solusi untuk masalah model salah prediksi meskipun dataset benar

PERUBAHAN KUNCI:
✅ freeze=0 → Train SEMUA layer (tidak freeze backbone)
✅ epochs=150 → Lebih lama untuk dataset custom
✅ patience=30 → Lebih sabar menunggu improvement
✅ lr0=0.001 → Learning rate lebih kecil untuk fine-tuning
✅ augment=True → Heavy augmentation untuk generalization
"""

import os
import logging
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import torch
import gc


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(log_dir: str, timestamp: str):
    """Setup logging untuk training"""
    log_path = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    return log_path


# ============================================================================
# EPOCH LOGGER CALLBACK
# ============================================================================

class TrainingLogger:
    """Callback untuk tracking progress per epoch"""
    
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.best_map50 = 0.0
        self.best_epoch = 0
        
    def _extract_loss(self, trainer):
        """Ekstrak nilai loss dari trainer"""
        try:
            if hasattr(trainer, 'loss') and trainer.loss is not None:
                if isinstance(trainer.loss, torch.Tensor):
                    return float(trainer.loss.item())
                return float(trainer.loss)
            
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                if isinstance(trainer.loss_items, dict):
                    return float(list(trainer.loss_items.values())[0])
                elif isinstance(trainer.loss_items, (list, tuple)):
                    return float(trainer.loss_items[0])
            
            if hasattr(trainer, 'tloss'):
                return float(trainer.tloss)
                
        except Exception:
            pass
        
        return None
    
    def on_train_epoch_end(self, trainer):
        """Hook dipanggil setiap akhir epoch"""
        epoch = trainer.epoch + 1
        elapsed = time.time() - self.start_time
        avg_time = elapsed / epoch
        eta_minutes = (avg_time * (self.total_epochs - epoch)) / 60
        
        msg_parts = [f"Epoch {epoch}/{self.total_epochs}"]
        
        loss = self._extract_loss(trainer)
        if loss is not None:
            msg_parts.append(f"Loss: {loss:.4f}")
        
        if hasattr(trainer, 'metrics') and trainer.metrics:
            m = trainer.metrics
            if hasattr(m, 'box'):
                current_map50 = m.box.map50
                msg_parts.extend([
                    f"mAP50: {current_map50:.4f}",
                    f"mAP95: {m.box.map:.4f}",
                    f"P: {m.box.mp:.4f}",
                    f"R: {m.box.mr:.4f}"
                ])
                
                if current_map50 > self.best_map50:
                    self.best_map50 = current_map50
                    self.best_epoch = epoch
                    msg_parts.append("⭐ NEW BEST!")
        
        msg_parts.append(f"⏱️ {avg_time:.1f}s/epoch")
        if eta_minutes > 0:
            msg_parts.append(f"ETA: {eta_minutes:.1f}m")
        
        logging.info(" | ".join(msg_parts))


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_system_info():
    """Dapatkan informasi sistem"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': None,
        'gpu_memory': None,
        'device': 'cpu'
    }
    
    if info['cuda_available']:
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info['device'] = 'cuda:0'
    
    return info


def cleanup_gpu_memory():
    """Cleanup GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def format_duration(seconds):
    """Format durasi ke format yang readable"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_training_summary(log_dir, model_name, dataset_yaml, config, metrics, duration, best_epoch):
    """Simpan ringkasan training ke file terpisah"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(log_dir, f"training_summary_{timestamp}.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("YOLOv11 TRAINING SUMMARY - FIXED VERSION\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-"*80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_yaml}\n")
        f.write(f"Epochs: {config['epochs']}\n")
        f.write(f"Image Size: {config['imgsz']}\n")
        f.write(f"Batch Size: {config['batch']}\n")
        f.write(f"Device: {config['device']}\n")
        f.write(f"Workers: {config['workers']}\n")
        f.write(f"Patience: {config['patience']}\n")
        f.write(f"Freeze Layers: {config.get('freeze', 'N/A')}\n")
        f.write(f"Learning Rate: {config.get('lr0', 'N/A')}\n")
        f.write(f"Weight Decay: {config.get('weight_decay', 'N/A')}\n\n")
        
        f.write("RESULTS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Training Duration: {format_duration(duration)}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"mAP50: {metrics.box.map50:.4f}\n")
        f.write(f"mAP50-95: {metrics.box.map:.4f}\n")
        f.write(f"Precision: {metrics.box.mp:.4f}\n")
        f.write(f"Recall: {metrics.box.mr:.4f}\n\n")
        
        f.write("="*80 + "\n")
    
    return summary_path


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model():
    """
    Training YOLOv11 dengan konfigurasi FIXED
    """
    
    # ========================================================================
    # CONFIGURATION - FIXED VERSION
    # ========================================================================
    
    dataset_yaml = os.path.join(os.getcwd(), "dataset.yaml")
    model_name = os.path.join("models", "yolo11n.pt")
    
    project_name = "navigation-assistance"
    experiment_name = "v2_fixed"  # v2 untuk versi perbaikan
    
    # 🔧 FIXED TRAINING PARAMETERS
    config = {
        'epochs': 150,          # ⬆️ Lebih lama dari 80 → 150
        'imgsz': 640,
        'batch': 8,
        'device': 0,
        'workers': 2,
        
        # 🎯 KEY CHANGES:
        'patience': 30,         # ⬆️ Dari 10 → 30 (lebih sabar)
        'freeze': 0,            # ⚡ CRITICAL: Dari 10 → 0 (train ALL layers)
        
        # Learning rate & optimization
        'lr0': 0.001,           # 🔽 Learning rate lebih kecil untuk stable learning
        'lrf': 0.01,            # Final lr = lr0 * lrf
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Data augmentation (lebih agresif)
        'hsv_h': 0.015,         # Hue augmentation
        'hsv_s': 0.7,           # Saturation
        'hsv_v': 0.4,           # Value
        'degrees': 10.0,        # Rotation
        'translate': 0.1,       # Translation
        'scale': 0.5,           # Scale
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,          # Vertical flip
        'fliplr': 0.5,          # Horizontal flip
        'mosaic': 1.0,          # Mosaic augmentation
        'mixup': 0.1,           # Mixup augmentation
        
        'seed': 42,
    }
    
    # Advanced settings
    use_cos_lr = True
    use_amp = True
    save_period = 20
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(project_name, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = setup_logger(log_dir, timestamp)
    sys_info = get_system_info()
    
    # ========================================================================
    # TRAINING INFO
    # ========================================================================
    
    logging.info("="*80)
    logging.info("🚀 YOLOv11 TRAINING - FIXED VERSION")
    logging.info("="*80)
    logging.info(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"📁 Project: {project_name}/{experiment_name}")
    logging.info("")
    
    logging.info("🔧 FIXES APPLIED:")
    logging.info("-"*80)
    logging.info("1. ⚡ freeze=0 → Train ALL layers (tidak freeze backbone)")
    logging.info("   Why? Dataset Anda berbeda dari COCO, model perlu belajar ulang")
    logging.info("")
    logging.info("2. ⏱️ epochs=150 (dari 80)")
    logging.info("   Why? Model butuh lebih banyak waktu untuk converge")
    logging.info("")
    logging.info("3. 🎯 patience=30 (dari 10)")
    logging.info("   Why? Training yang lebih dalam butuh patience lebih")
    logging.info("")
    logging.info("4. 📉 lr0=0.001 (learning rate lebih kecil)")
    logging.info("   Why? Stable learning untuk fine-tuning semua layer")
    logging.info("")
    logging.info("5. 🎨 Heavy augmentation")
    logging.info("   Why? Meningkatkan generalization & robustness")
    logging.info("")
    
    logging.info("CONFIGURATION:")
    logging.info("-"*80)
    logging.info(f"Model: {model_name}")
    logging.info(f"Dataset: {dataset_yaml}")
    logging.info(f"Epochs: {config['epochs']}")
    logging.info(f"Image Size: {config['imgsz']}")
    logging.info(f"Batch Size: {config['batch']}")
    logging.info(f"Freeze Layers: {config['freeze']} ⚡ (ALL LAYERS TRAINABLE)")
    logging.info(f"Learning Rate: {config['lr0']}")
    logging.info(f"Patience: {config['patience']}")
    logging.info("")
    
    logging.info("SYSTEM INFO:")
    logging.info("-"*80)
    if sys_info['cuda_available']:
        logging.info(f"GPU: {sys_info['gpu_name']} ({sys_info['gpu_memory']:.1f} GB)")
    else:
        logging.info("GPU: Not available (using CPU)")
    logging.info("")
    
    logging.info("="*80)
    logging.info("")
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    
    try:
        training_start = time.time()
        
        logging.info("📦 Loading model...")
        model = YOLO(model_name)
        
        epoch_logger = TrainingLogger(config['epochs'])
        model.add_callback("on_train_epoch_end", epoch_logger.on_train_epoch_end)
        
        logging.info("🏋️ Starting training...\n")
        logging.info("-"*80)
        
        results = model.train(
            data=dataset_yaml,
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            device=config['device'],
            workers=config['workers'],
            patience=config['patience'],
            freeze=config['freeze'],
            
            # Optimizer params
            lr0=config['lr0'],
            lrf=config['lrf'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
            
            # Augmentation params
            hsv_h=config['hsv_h'],
            hsv_s=config['hsv_s'],
            hsv_v=config['hsv_v'],
            degrees=config['degrees'],
            translate=config['translate'],
            scale=config['scale'],
            shear=config['shear'],
            perspective=config['perspective'],
            flipud=config['flipud'],
            fliplr=config['fliplr'],
            mosaic=config['mosaic'],
            mixup=config['mixup'],
            
            # Other params
            cos_lr=use_cos_lr,
            amp=use_amp,
            project=project_name,
            name=experiment_name,
            exist_ok=False,
            pretrained=True,
            verbose=False,
            cache=False,
            resume=False,
            save_period=save_period,
            seed=config['seed'],
        )
        
        training_duration = time.time() - training_start
        
        # ====================================================================
        # VALIDATION
        # ====================================================================
        
        logging.info("-"*80)
        logging.info("\n📊 Running final validation...")
        metrics = model.val()
        
        # ====================================================================
        # RESULTS
        # ====================================================================
        
        logging.info("")
        logging.info("="*80)
        logging.info("✅ TRAINING COMPLETED!")
        logging.info("="*80)
        logging.info(f"⏱️ Duration: {format_duration(training_duration)}")
        logging.info(f"🏆 Best Epoch: {epoch_logger.best_epoch}/{config['epochs']}")
        logging.info("")
        
        logging.info("FINAL METRICS:")
        logging.info("-"*80)
        logging.info(f"mAP50:     {metrics.box.map50:.4f}")
        logging.info(f"mAP50-95:  {metrics.box.map:.4f}")
        logging.info(f"Precision: {metrics.box.mp:.4f}")
        logging.info(f"Recall:    {metrics.box.mr:.4f}")
        logging.info("")
        
        logging.info("SAVED FILES:")
        logging.info("-"*80)
        logging.info(f"📁 Model directory: {results.save_dir}")
        logging.info(f"🏆 Best weights: {Path(results.save_dir) / 'weights' / 'best.pt'}")
        logging.info(f"📄 Last weights: {Path(results.save_dir) / 'weights' / 'last.pt'}")
        logging.info(f"📝 Training log: {log_path}")
        
        summary_path = save_training_summary(
            log_dir, model_name, dataset_yaml, config, 
            metrics, training_duration, epoch_logger.best_epoch
        )
        logging.info(f"📋 Summary: {summary_path}")
        
        logging.info("")
        logging.info("🎯 NEXT STEPS:")
        logging.info("-"*80)
        logging.info("1. Test model dengan: python selective_test_by_filename.py")
        logging.info("2. Convert to TFLite: python convert_to_tflite.py")
        logging.info("3. Compare dengan model lama untuk lihat improvement")
        
        logging.info("="*80)
        
        cleanup_gpu_memory()
        
        return results, metrics
        
    except KeyboardInterrupt:
        logging.warning("\n⚠️ Training interrupted by user")
        cleanup_gpu_memory()
        raise
        
    except Exception as e:
        logging.error(f"\n❌ Training failed: {str(e)}")
        cleanup_gpu_memory()
        raise


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("🔧 FIXED TRAINING CONFIGURATION")
        print("="*80)
        print("\nKEY CHANGES:")
        print("  ⚡ freeze=0 (train ALL layers)")
        print("  ⏱️ epochs=150 (more training time)")
        print("  🎯 patience=30 (more patience)")
        print("  📉 lr0=0.001 (smaller learning rate)")
        print("  🎨 Heavy augmentation")
        print("\nExpected improvement:")
        print("  ✅ Chair detected as chair (not person)")
        print("  ✅ Door detected as door (not chair)")
        print("  ✅ Table detected as table (not stair)")
        print("\n" + "="*80)
        
        input("\nPress ENTER to start training...")
        
        results, metrics = train_model()
        print("\n✨ Training berhasil! Lihat log untuk detail lengkap.")
        
    except Exception as e:
        print(f"\n❌ Training gagal: {str(e)}")