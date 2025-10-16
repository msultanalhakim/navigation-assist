"""
Random Search Hyperparameter YOLOv11 - Improved Version
Tujuan: Eksplorasi acak kombinasi hyperparameter dengan struktur direktori rapi
"""

import os
import random
import csv
import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import multiprocessing
import torch


# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

def setup_directories(base_dir: str, timestamp: str):
    """
    Setup struktur direktori yang rapi:
    
    hyperparameter_tuning/
    ├── random_search/
    │   ├── runs/
    │   │   └── YYYYMMDD_HHMMSS/
    │   │       ├── exp001/
    │   │       ├── exp002/
    │   │       └── ...
    │   ├── logs/
    │   │   └── random_search_YYYYMMDD_HHMMSS.log
    │   └── results/
    │       ├── results_YYYYMMDD_HHMMSS.csv
    │       └── config_YYYYMMDD_HHMMSS.json
    """
    
    # Main directories
    random_dir = Path(base_dir) / "random_search"
    runs_dir = random_dir / "runs" / timestamp
    logs_dir = random_dir / "logs"
    results_dir = random_dir / "results"
    
    # Create all directories
    for directory in [runs_dir, logs_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    return {
        'base': random_dir,
        'runs': runs_dir,
        'logs': logs_dir,
        'results': results_dir
    }


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(logs_dir: Path, timestamp: str):
    """Setup logging dengan format yang lebih informatif"""
    log_path = logs_dir / f"random_search_{timestamp}.log"
    
    # Clear existing handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler dengan encoding UTF-8
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter dengan info lebih detail
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
# SYSTEM INFO
# ============================================================================

def log_system_info():
    """Log informasi sistem untuk debugging"""
    logging.info("=" * 80)
    logging.info("SYSTEM INFORMATION")
    logging.info("=" * 80)
    
    # Python version
    logging.info(f"Python: {sys.version.split()[0]}")
    
    # CPU
    cpu_count = multiprocessing.cpu_count()
    logging.info(f"CPU Cores: {cpu_count}")
    
    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"GPU: {gpu_name}")
        logging.info(f"GPU Memory: {gpu_memory:.2f} GB")
        logging.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logging.info("GPU: Not available (CPU mode)")
        logging.warning("⚠️ Training akan lambat tanpa GPU!")
    
    # PyTorch
    logging.info(f"PyTorch: {torch.__version__}")
    
    try:
        from ultralytics import __version__
        logging.info(f"Ultralytics: {__version__}")
    except:
        logging.warning("⚠️ Tidak dapat membaca versi Ultralytics")
    
    logging.info("=" * 80)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config(dataset_yaml: str, model_path: str) -> bool:
    """Validasi file-file yang diperlukan"""
    logging.info("Validating configuration...")
    
    errors = []
    
    # Check dataset.yaml
    if not os.path.exists(dataset_yaml):
        errors.append(f"Dataset YAML tidak ditemukan: {dataset_yaml}")
    else:
        logging.info(f"✓ Dataset YAML found: {dataset_yaml}")
    
    # Check model
    if not os.path.exists(model_path):
        errors.append(f"Model tidak ditemukan: {model_path}")
    else:
        logging.info(f"✓ Model found: {model_path}")
    
    # Check CUDA
    if not torch.cuda.is_available():
        logging.warning("⚠️ CUDA tidak tersedia - training akan sangat lambat!")
    
    if errors:
        logging.error("=" * 80)
        logging.error("VALIDATION FAILED")
        logging.error("=" * 80)
        for error in errors:
            logging.error(f"✗ {error}")
        return False
    
    logging.info("✓ Validation passed")
    return True


# ============================================================================
# SAVE CONFIGURATION
# ============================================================================

def save_config(results_dir: Path, timestamp: str, config: dict):
    """Simpan konfigurasi eksperimen untuk reproducibility"""
    config_path = results_dir / f"config_{timestamp}.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"Configuration saved: {config_path}")
    return config_path


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def run_random_search():
    """
    Random search untuk eksplorasi hyperparameter space YOLOv11
    """
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    CONFIG = {
        # Paths
        'dataset_yaml': os.path.join(os.getcwd(), "dataset.yaml"),
        'model_path': os.path.join("models", "yolo11n.pt"),
        'base_dir': "hyperparameter_tuning",
        
        # Search space
        'lr_range': (0.001, 0.01),
        'momentum_range': (0.85, 0.95),
        'weight_decay_range': (0.0001, 0.001),
        'batch_options': [8, 16],
        'optimizer_options': ["SGD", "AdamW"],
        
        # Training
        'n_experiments': 10,
        'epochs': 30,
        'img_size': 640,
        'patience': 10,
        'workers': min(4, multiprocessing.cpu_count() - 1),
        
        # Device
        'device': 0 if torch.cuda.is_available() else 'cpu'
    }
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup directories
    dirs = setup_directories(CONFIG['base_dir'], timestamp)
    
    # Setup logging
    log_path = setup_logger(dirs['logs'], timestamp)
    
    logging.info("=" * 80)
    logging.info("🔬 YOLOv11 RANDOM SEARCH HYPERPARAMETER TUNING")
    logging.info("=" * 80)
    logging.info(f"Timestamp: {timestamp}")
    logging.info(f"Log file: {log_path}")
    logging.info("")
    
    # Log system info
    log_system_info()
    
    # Validate configuration
    if not validate_config(CONFIG['dataset_yaml'], CONFIG['model_path']):
        logging.error("Exiting due to validation errors")
        return
    
    # Save configuration
    config_path = save_config(dirs['results'], timestamp, CONFIG)
    
    # ========================================================================
    # SEARCH SPACE INFO
    # ========================================================================
    
    logging.info("")
    logging.info("=" * 80)
    logging.info("RANDOM SEARCH CONFIGURATION")
    logging.info("=" * 80)
    logging.info(f"Number of experiments: {CONFIG['n_experiments']}")
    logging.info(f"Epochs per experiment: {CONFIG['epochs']}")
    logging.info(f"Image size: {CONFIG['img_size']}")
    logging.info(f"Workers: {CONFIG['workers']}")
    logging.info(f"Device: {CONFIG['device']}")
    logging.info("")
    logging.info("Search Space:")
    logging.info(f"  • Learning Rate    : {CONFIG['lr_range']}")
    logging.info(f"  • Momentum         : {CONFIG['momentum_range']}")
    logging.info(f"  • Weight Decay     : {CONFIG['weight_decay_range']}")
    logging.info(f"  • Batch Size       : {CONFIG['batch_options']}")
    logging.info(f"  • Optimizer        : {CONFIG['optimizer_options']}")
    logging.info("=" * 80)
    
    # ========================================================================
    # CSV SETUP
    # ========================================================================
    
    csv_path = dirs['results'] / f"results_{timestamp}.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'exp', 'lr0', 'momentum', 'weight_decay', 'batch', 'optimizer',
            'map50', 'map95', 'precision', 'recall', 'status', 'duration_min',
            'run_dir'
        ])
    
    logging.info(f"Results CSV: {csv_path}")
    logging.info("=" * 80)
    
    # ========================================================================
    # RANDOM SEARCH LOOP
    # ========================================================================
    
    import time
    success_count = 0
    start_time = time.time()
    
    for i in range(1, CONFIG['n_experiments'] + 1):
        exp_start = time.time()
        
        # Generate random hyperparameters
        lr0 = round(random.uniform(*CONFIG['lr_range']), 5)
        momentum = round(random.uniform(*CONFIG['momentum_range']), 4)
        weight_decay = round(random.uniform(*CONFIG['weight_decay_range']), 6)
        batch = random.choice(CONFIG['batch_options'])
        optimizer = random.choice(CONFIG['optimizer_options'])
        
        exp_name = f"exp{i:03d}"
        
        # --------------------------------------------------------------------
        # LOG EXPERIMENT START
        # --------------------------------------------------------------------
        
        logging.info("")
        logging.info("=" * 80)
        logging.info(f"🧪 EXPERIMENT {i}/{CONFIG['n_experiments']}")
        logging.info("=" * 80)
        logging.info(f"Hyperparameters:")
        logging.info(f"  • Learning Rate   : {lr0}")
        logging.info(f"  • Momentum        : {momentum}")
        logging.info(f"  • Weight Decay    : {weight_decay}")
        logging.info(f"  • Batch Size      : {batch}")
        logging.info(f"  • Optimizer       : {optimizer}")
        logging.info(f"Run directory: {exp_name}")
        logging.info("-" * 80)
        
        # Init result variables
        status = "FAILED"
        map50 = map95 = precision = recall = 0.0
        run_dir = ""
        
        try:
            # ----------------------------------------------------------------
            # TRAINING
            # ----------------------------------------------------------------
            
            logging.info(f"Loading model: {CONFIG['model_path']}")
            model = YOLO(CONFIG['model_path'])
            
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
                project=str(dirs['runs']),
                name=exp_name,
                exist_ok=False,
                verbose=True,
                plots=True,
                save=True,
                cache=False
            )
            
            # ----------------------------------------------------------------
            # VALIDATION
            # ----------------------------------------------------------------
            
            logging.info("Running validation...")
            metrics = model.val()
            
            # Extract metrics
            map50 = float(metrics.box.map50)
            map95 = float(metrics.box.map)
            precision = float(metrics.box.mp)
            recall = float(metrics.box.mr)
            
            status = "SUCCESS"
            success_count += 1
            
            # Get run directory
            run_dir = str(dirs['runs'] / exp_name)
            
            # ----------------------------------------------------------------
            # LOG RESULTS
            # ----------------------------------------------------------------
            
            duration = (time.time() - exp_start) / 60
            
            logging.info("-" * 80)
            logging.info(f"✅ EXPERIMENT {i} COMPLETED ({duration:.1f} min)")
            logging.info(f"Results:")
            logging.info(f"  • mAP50     : {map50:.4f}")
            logging.info(f"  • mAP95     : {map95:.4f}")
            logging.info(f"  • Precision : {precision:.4f}")
            logging.info(f"  • Recall    : {recall:.4f}")
            logging.info("=" * 80)
            
        except KeyboardInterrupt:
            logging.warning(f"⚠️ Experiment {i} interrupted by user")
            status = "INTERRUPTED"
            break
            
        except RuntimeError as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                logging.error(f"❌ CUDA Out of Memory")
                logging.error(f"   Suggestion: Reduce batch size (current: {batch})")
                status = "OOM"
            else:
                logging.error(f"❌ Runtime Error: {str(e)}")
                logging.debug(f"Full traceback:", exc_info=True)
                status = "RUNTIME_ERROR"
        
        except Exception as e:
            logging.error(f"❌ Unexpected Error: {str(e)}")
            logging.debug("Full traceback:", exc_info=True)
            status = "ERROR"
        
        finally:
            # ----------------------------------------------------------------
            # SAVE TO CSV
            # ----------------------------------------------------------------
            
            duration = (time.time() - exp_start) / 60
            
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    i, lr0, momentum, weight_decay, batch, optimizer,
                    f"{map50:.4f}", f"{map95:.4f}", f"{precision:.4f}", 
                    f"{recall:.4f}", status, f"{duration:.2f}", run_dir
                ])
            
            # Cleanup
            try:
                del model
            except:
                pass
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    total_duration = (time.time() - start_time) / 3600
    
    logging.info("")
    logging.info("=" * 80)
    logging.info("🎉 RANDOM SEARCH COMPLETED")
    logging.info("=" * 80)
    logging.info(f"✅ Successful    : {success_count}/{CONFIG['n_experiments']}")
    logging.info(f"❌ Failed        : {CONFIG['n_experiments'] - success_count}/{CONFIG['n_experiments']}")
    logging.info(f"⏱️  Total time    : {total_duration:.2f} hours")
    logging.info(f"📊 Results CSV   : {csv_path}")
    logging.info(f"📝 Log file      : {log_path}")
    logging.info(f"⚙️  Configuration : {config_path}")
    
    # Find best result
    if success_count > 0:
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            df_success = df[df['status'] == 'SUCCESS']
            
            if len(df_success) > 0:
                best = df_success.loc[df_success['map50'].idxmax()]
                
                logging.info("")
                logging.info("🏆 BEST RESULT:")
                logging.info(f"  Experiment    : #{int(best['exp'])}")
                logging.info(f"  mAP50         : {best['map50']}")
                logging.info(f"  Hyperparameters:")
                logging.info(f"    • lr0          : {best['lr0']}")
                logging.info(f"    • momentum     : {best['momentum']}")
                logging.info(f"    • weight_decay : {best['weight_decay']}")
                logging.info(f"    • batch        : {int(best['batch'])}")
                logging.info(f"    • optimizer    : {best['optimizer']}")
                logging.info(f"  Directory     : {best['run_dir']}")
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
        run_random_search()
    except KeyboardInterrupt:
        logging.warning("\n⚠️ Program dihentikan oleh user")
    except Exception as e:
        logging.error(f"\n❌ Fatal error: {str(e)}")
        logging.debug("Full traceback:", exc_info=True)
        sys.exit(1)