"""
Grid Search Hyperparameter YOLOv11 - Improved & Clean Version
Fokus: Eksplorasi sistematis dengan struktur direktori rapi dan debugging lengkap
"""

import os
import csv
import logging
import shutil
import time
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from ultralytics import YOLO
import multiprocessing
import torch
import gc


# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

def setup_directories(base_dir: str, timestamp: str):
    """
    Setup struktur direktori yang rapi:
    
    hyperparameter_tuning/
    ├── grid_search/
    │   ├── runs/
    │   │   └── YYYYMMDD_HHMMSS/
    │   │       ├── grid001/
    │   │       ├── grid002/
    │   │       └── ...
    │   ├── logs/
    │   │   └── grid_search_YYYYMMDD_HHMMSS.log
    │   └── results/
    │       ├── results_YYYYMMDD_HHMMSS.csv
    │       └── config_YYYYMMDD_HHMMSS.json
    """
    
    grid_dir = Path(base_dir) / "grid_search"
    runs_dir = grid_dir / "runs" / timestamp
    logs_dir = grid_dir / "logs"
    results_dir = grid_dir / "results"
    
    for directory in [runs_dir, logs_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    return {
        'base': grid_dir,
        'runs': runs_dir,
        'logs': logs_dir,
        'results': results_dir
    }


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(logs_dir: Path, timestamp: str):
    """Setup logging dengan format informatif"""
    log_path = logs_dir / f"grid_search_{timestamp}.log"
    
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)
    
    return log_path


# ============================================================================
# EPOCH LOGGER CALLBACK
# ============================================================================

class EpochLogger:
    """Callback untuk tracking progress per epoch dengan debugging"""
    
    def __init__(self, grid_num, total_epochs):
        self.grid_num = grid_num
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.epoch_times = []
        
    def _extract_loss(self, trainer):
        """Ekstrak nilai loss dengan fallback bertingkat"""
        try:
            if hasattr(trainer, 'loss') and trainer.loss is not None:
                if isinstance(trainer.loss, torch.Tensor):
                    return float(trainer.loss.item())
                return float(trainer.loss)
            
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                if isinstance(trainer.loss_items, dict):
                    total_loss = sum(trainer.loss_items.values())
                    return float(total_loss)
                elif isinstance(trainer.loss_items, (list, tuple)):
                    return float(sum(trainer.loss_items))
            
            if hasattr(trainer, 'tloss'):
                return float(trainer.tloss)
                
        except Exception as e:
            logging.debug(f"Error extracting loss: {str(e)}")
        
        return None
    
    def on_train_epoch_end(self, trainer):
        """Hook dipanggil YOLO setiap akhir epoch"""
        epoch = trainer.epoch + 1
        elapsed = time.time() - self.start_time
        
        self.epoch_times.append(elapsed / epoch)
        avg_time = sum(self.epoch_times) / len(self.epoch_times)
        eta_seconds = avg_time * (self.total_epochs - epoch)
        eta_minutes = eta_seconds / 60
        
        msg_parts = [f"[Grid {self.grid_num}] Epoch {epoch}/{self.total_epochs}"]
        
        loss = self._extract_loss(trainer)
        if loss is not None:
            msg_parts.append(f"Loss: {loss:.4f}")
        else:
            msg_parts.append("Loss: N/A")
        
        if hasattr(trainer, 'metrics') and trainer.metrics:
            m = trainer.metrics
            if hasattr(m, 'box'):
                msg_parts.extend([
                    f"mAP50: {m.box.map50:.4f}",
                    f"mAP95: {m.box.map:.4f}"
                ])
        
        msg_parts.append(f"⏱️ {avg_time:.1f}s/epoch")
        msg_parts.append(f"ETA: {eta_minutes:.1f}m")
        
        logging.info(" | ".join(msg_parts))


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_optimal_workers():
    """Tentukan jumlah workers optimal"""
    cpu_count = multiprocessing.cpu_count()
    
    if cpu_count <= 4:
        return 2
    elif cpu_count <= 8:
        return 4
    else:
        return min(6, cpu_count - 2)


def cleanup_failed_experiment(runs_dir: Path, grid_name: str):
    """Hapus folder eksperimen yang gagal"""
    try:
        exp_path = runs_dir / grid_name
        if not exp_path.exists():
            return
        
        weights_dir = exp_path / "weights"
        has_weights = weights_dir.exists() and any(weights_dir.glob("*.pt"))
        
        if not has_weights:
            logging.warning(f"🗑️ Cleaning up failed experiment: {grid_name}")
            shutil.rmtree(exp_path)
    except Exception as e:
        logging.error(f"Failed to cleanup: {str(e)}")


def release_gpu_memory():
    """Force GPU memory release"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def format_time(seconds):
    """Format detik ke HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def log_system_info():
    """Log informasi sistem"""
    logging.info("=" * 80)
    logging.info("SYSTEM INFORMATION")
    logging.info("=" * 80)
    
    logging.info(f"Python: {sys.version.split()[0]}")
    logging.info(f"CPU Cores: {multiprocessing.cpu_count()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"GPU: {gpu_name}")
        logging.info(f"GPU Memory: {gpu_memory:.2f} GB")
        logging.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logging.info("GPU: Not available (CPU mode)")
        logging.warning("⚠️ Training akan sangat lambat!")
    
    logging.info(f"PyTorch: {torch.__version__}")
    
    try:
        from ultralytics import __version__
        logging.info(f"Ultralytics: {__version__}")
    except:
        pass
    
    logging.info("=" * 80)


def validate_config(dataset_yaml: str, model_path: str) -> bool:
    """Validasi konfigurasi"""
    logging.info("Validating configuration...")
    
    errors = []
    
    if not os.path.exists(dataset_yaml):
        errors.append(f"Dataset YAML not found: {dataset_yaml}")
    else:
        logging.info(f"✓ Dataset YAML: {dataset_yaml}")
    
    if not os.path.exists(model_path):
        errors.append(f"Model not found: {model_path}")
    else:
        logging.info(f"✓ Model: {model_path}")
    
    if errors:
        logging.error("=" * 80)
        logging.error("VALIDATION FAILED")
        logging.error("=" * 80)
        for error in errors:
            logging.error(f"✗ {error}")
        return False
    
    logging.info("✓ Validation passed")
    return True


def save_config(results_dir: Path, timestamp: str, config: dict):
    """Simpan konfigurasi eksperimen"""
    config_path = results_dir / f"config_{timestamp}.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"Configuration saved: {config_path}")
    return config_path


# ============================================================================
# MAIN GRID SEARCH
# ============================================================================

def run_grid_search():
    """Grid search untuk eksplorasi sistematis hyperparameter YOLOv11"""
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    CONFIG = {
        'dataset_yaml': os.path.join(os.getcwd(), "dataset.yaml"),
        'model_path': os.path.join("models", "yolo11n.pt"),
        'base_dir': "hyperparameter_tuning",
        
        'lr_values': [0.003, 0.005, 0.007],
        'momentum_values': [0.88, 0.9, 0.92],
        'weight_decay_values': [0.0003, 0.0005],
        'batch_values': [8, 16],
        'optimizer_values': ["SGD", "AdamW"],
        
        'epochs': 30,
        'img_size': 640,
        'patience': 10,
        'workers': get_optimal_workers(),
        
        'device': 0 if torch.cuda.is_available() else 'cpu'
    }
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirs = setup_directories(CONFIG['base_dir'], timestamp)
    log_path = setup_logger(dirs['logs'], timestamp)
    
    logging.info("=" * 80)
    logging.info("🔬 YOLOv11 GRID SEARCH HYPERPARAMETER TUNING")
    logging.info("=" * 80)
    logging.info(f"Timestamp: {timestamp}")
    logging.info(f"Log file: {log_path}")
    logging.info("")
    
    log_system_info()
    
    if not validate_config(CONFIG['dataset_yaml'], CONFIG['model_path']):
        logging.error("Exiting due to validation errors")
        return
    
    config_path = save_config(dirs['results'], timestamp, CONFIG)
    
    # ========================================================================
    # GENERATE COMBINATIONS
    # ========================================================================
    
    combinations = list(product(
        CONFIG['lr_values'],
        CONFIG['momentum_values'],
        CONFIG['weight_decay_values'],
        CONFIG['batch_values'],
        CONFIG['optimizer_values']
    ))
    
    total_combinations = len(combinations)
    
    # ========================================================================
    # GRID SEARCH INFO
    # ========================================================================
    
    logging.info("")
    logging.info("=" * 80)
    logging.info("GRID SEARCH CONFIGURATION")
    logging.info("=" * 80)
    logging.info(f"Total combinations: {total_combinations}")
    logging.info(f"Epochs per combination: {CONFIG['epochs']}")
    logging.info(f"Image size: {CONFIG['img_size']}")
    logging.info(f"Workers: {CONFIG['workers']}")
    logging.info(f"Device: {CONFIG['device']}")
    logging.info("")
    logging.info("Grid Search Space:")
    logging.info(f"  • Learning Rate    : {CONFIG['lr_values']}")
    logging.info(f"  • Momentum         : {CONFIG['momentum_values']}")
    logging.info(f"  • Weight Decay     : {CONFIG['weight_decay_values']}")
    logging.info(f"  • Batch Size       : {CONFIG['batch_values']}")
    logging.info(f"  • Optimizer        : {CONFIG['optimizer_values']}")
    logging.info("=" * 80)
    
    # ========================================================================
    # CSV SETUP
    # ========================================================================
    
    csv_path = dirs['results'] / f"results_{timestamp}.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'grid', 'lr0', 'momentum', 'weight_decay', 'batch', 'optimizer',
            'map50', 'map95', 'precision', 'recall', 'status', 'duration_min',
            'run_dir'
        ])
    
    logging.info(f"Results CSV: {csv_path}")
    logging.info("=" * 80)
    
    # ========================================================================
    # GRID SEARCH LOOP
    # ========================================================================
    
    success_count = 0
    global_start = time.time()
    
    for i, (lr0, momentum, weight_decay, batch, optimizer) in enumerate(combinations, 1):
        grid_start = time.time()
        
        progress_pct = (i / total_combinations) * 100
        elapsed_total = time.time() - global_start
        avg_time_per_grid = elapsed_total / i if i > 0 else 0
        eta_total = avg_time_per_grid * (total_combinations - i)
        
        grid_name = f"grid{i:03d}"
        
        logging.info("")
        logging.info("=" * 80)
        logging.info(f"🧪 GRID {i}/{total_combinations} ({progress_pct:.1f}%)")
        logging.info("=" * 80)
        logging.info(f"Hyperparameters:")
        logging.info(f"  • Learning Rate   : {lr0}")
        logging.info(f"  • Momentum        : {momentum}")
        logging.info(f"  • Weight Decay    : {weight_decay}")
        logging.info(f"  • Batch Size      : {batch}")
        logging.info(f"  • Optimizer       : {optimizer}")
        logging.info(f"Progress: {i-1} completed | ETA: {format_time(eta_total)}")
        logging.info(f"Save directory: {grid_name}")
        logging.info("-" * 80)
        
        status = "FAILED"
        map50 = map95 = precision = recall = 0.0
        run_dir = ""
        
        try:
            logging.info(f"Loading model: {CONFIG['model_path']}")
            model = YOLO(CONFIG['model_path'])
            
            epoch_logger = EpochLogger(i, CONFIG['epochs'])
            model.add_callback("on_train_epoch_end", epoch_logger.on_train_epoch_end)
            
            logging.info("Starting training...")
            results = model.train(
                data=CONFIG['dataset_yaml'],
                epochs=CONFIG['epochs'],
                imgsz=CONFIG['img_size'],
                batch=batch,
                lr0=lr0,
                momentum=momentum,
                weight_decay=weight_decay,
                optimizer=optimizer,
                device=CONFIG['device'],
                workers=CONFIG['workers'],
                patience=CONFIG['patience'],
                deterministic=False,
                project=str(dirs['runs']),
                name=grid_name,
                exist_ok=False,
                pretrained=True,
                verbose=False,
                cache=False,
                resume=False,
            )
            
            logging.info(f"[Grid {i}] Running validation...")
            metrics = model.val()
            
            map50 = float(metrics.box.map50)
            map95 = float(metrics.box.map)
            precision = float(metrics.box.mp)
            recall = float(metrics.box.mr)
            
            status = "SUCCESS"
            success_count += 1
            run_dir = str(dirs['runs'] / grid_name)
            
            duration = (time.time() - grid_start) / 60
            
            logging.info("-" * 80)
            logging.info(f"✅ GRID {i} COMPLETED ({duration:.1f} min)")
            logging.info(f"Results:")
            logging.info(f"  • mAP50     : {map50:.4f}")
            logging.info(f"  • mAP95     : {map95:.4f}")
            logging.info(f"  • Precision : {precision:.4f}")
            logging.info(f"  • Recall    : {recall:.4f}")
            logging.info("=" * 80)
            
        except KeyboardInterrupt:
            logging.warning(f"⚠️ Grid {i} interrupted by user")
            status = "INTERRUPTED"
            cleanup_failed_experiment(dirs['runs'], grid_name)
            break
            
        except RuntimeError as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                logging.error(f"❌ [Grid {i}] CUDA Out of Memory")
                logging.error(f"   Try reducing batch size (current: {batch})")
                status = "OOM"
            else:
                logging.error(f"❌ [Grid {i}] Runtime Error: {str(e)}")
                logging.debug("Full traceback:", exc_info=True)
                status = "RUNTIME_ERROR"
            
            cleanup_failed_experiment(dirs['runs'], grid_name)
        
        except Exception as e:
            logging.error(f"❌ [Grid {i}] Unexpected Error: {str(e)}")
            logging.debug("Full traceback:", exc_info=True)
            status = "ERROR"
            cleanup_failed_experiment(dirs['runs'], grid_name)
        
        finally:
            duration = (time.time() - grid_start) / 60
            
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    i, lr0, momentum, weight_decay, batch, optimizer,
                    f"{map50:.4f}", f"{map95:.4f}", f"{precision:.4f}",
                    f"{recall:.4f}", status, f"{duration:.2f}", run_dir
                ])
            
            try:
                del model
            except:
                pass
            
            release_gpu_memory()
            logging.info(f"🧹 Memory cleaned | Next grid in 2s...\n")
            time.sleep(2)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    total_duration = (time.time() - global_start) / 3600
    
    logging.info("")
    logging.info("=" * 80)
    logging.info("🎉 GRID SEARCH COMPLETED")
    logging.info("=" * 80)
    logging.info(f"✅ Successful : {success_count}/{total_combinations}")
    logging.info(f"❌ Failed     : {total_combinations - success_count}/{total_combinations}")
    logging.info(f"⏱️  Total time : {total_duration:.2f} hours")
    logging.info(f"📊 Results    : {csv_path}")
    logging.info(f"📝 Log        : {log_path}")
    logging.info(f"⚙️  Config     : {config_path}")
    
    if success_count > 0:
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            df_success = df[df['status'] == 'SUCCESS']
            
            if len(df_success) > 0:
                best = df_success.loc[df_success['map50'].idxmax()]
                
                logging.info("")
                logging.info("🏆 BEST RESULT:")
                logging.info(f"  Grid          : #{int(best['grid'])}")
                logging.info(f"  mAP50         : {best['map50']}")
                logging.info(f"  Hyperparameters:")
                logging.info(f"    • lr0          : {best['lr0']}")
                logging.info(f"    • momentum     : {best['momentum']}")
                logging.info(f"    • weight_decay : {best['weight_decay']}")
                logging.info(f"    • batch        : {int(best['batch'])}")
                logging.info(f"    • optimizer    : {best['optimizer']}")
                logging.info(f"  Directory     : {best['run_dir']}")
                
                logging.info("")
                logging.info("📊 TOP 5 RESULTS:")
                top5 = df_success.nlargest(5, 'map50')
                for idx, row in top5.iterrows():
                    logging.info(f"  #{int(row['grid'])}: mAP50={row['map50']} | "
                               f"lr0={row['lr0']} | batch={int(row['batch'])} | "
                               f"opt={row['optimizer']}")
        except ImportError:
            logging.info("\n💡 Install pandas untuk analisis lebih detail: pip install pandas")
        except Exception as e:
            logging.warning(f"Tidak dapat membaca best result: {str(e)}")
    
    logging.info("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    try:
        run_grid_search()
    except KeyboardInterrupt:
        logging.warning("\n⚠️ Program dihentikan oleh user")
    except Exception as e:
        logging.error(f"\n❌ Fatal error: {str(e)}")
        logging.debug("Full traceback:", exc_info=True)
        sys.exit(1)