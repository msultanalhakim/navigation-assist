"""
Unified Hyperparameter Search for YOLOv11
Supports both Grid Search and Random Search
"""

from ultralytics import YOLO
import random
import os
import csv
import time
import json
import shutil
from datetime import datetime
from itertools import product
import torch
import argparse

from scripts.utils import logger, config


class HyperparameterSearch:
    """Hyperparameter search manager"""
    
    def __init__(self, search_type='random', n_experiments=10):
        self.search_type = search_type
        self.n_experiments = n_experiments
        
        # Load from config
        hp_config = config.get('hyperparameter_search', {})
        
        # Common settings
        self.epochs = hp_config.get('epochs', 40)
        self.imgsz = hp_config.get('imgsz', 640)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.workers = hp_config.get('workers', 2)
        self.amp = hp_config.get('amp', True)
        self.save_period = hp_config.get('save_period', 10)
        self.patience = hp_config.get('patience', 10)
        
        # Search space
        self.search_space = hp_config.get('search_space', {
            "lr0": (0.005, 0.02),
            "momentum": (0.90, 0.945),
            "weight_decay": (0.0003, 0.0012),
            "batch": [16, 24, 32],
            "optimizer": ["SGD", "AdamW"],
            "cos_lr": [True, False],
        })
        
        # Setup directories
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("runs", f"{search_type}_search", ts)
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.run_dir, f"results_{ts}.csv")
        self.todo_json = os.path.join(self.run_dir, "todo.json")
        
    def generate_experiments(self):
        """Generate experiment configurations"""
        if self.search_type == 'grid':
            return self._generate_grid()
        else:
            return self._generate_random()
    
    def _generate_grid(self):
        """Generate grid search combinations"""
        # Convert search space to grid format
        grid_space = {}
        for key, value in self.search_space.items():
            if isinstance(value, tuple):
                # Convert range to discrete values
                if key == 'lr0':
                    grid_space[key] = [0.007, 0.009, 0.011]
                elif key == 'momentum':
                    grid_space[key] = [0.90, 0.93]
                elif key == 'weight_decay':
                    grid_space[key] = [0.0003, 0.0005]
            else:
                grid_space[key] = value
        
        combinations = list(product(
            grid_space["lr0"],
            grid_space["momentum"],
            grid_space["weight_decay"],
            grid_space["batch"],
            grid_space["optimizer"],
            grid_space["cos_lr"]
        ))
        
        experiments = []
        for i, (lr0, momentum, weight_decay, batch, optimizer, cos_lr) in enumerate(combinations, 1):
            experiments.append({
                "name": f"grid{i:03d}",
                "lr0": lr0,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "batch": batch,
                "optimizer": optimizer,
                "cos_lr": cos_lr,
            })
        
        return experiments
    
    def _generate_random(self):
        """Generate random search combinations"""
        experiments = []
        for i in range(1, self.n_experiments + 1):
            cfg = {
                "name": f"exp{i:03d}",
                "lr0": round(random.uniform(*self.search_space["lr0"]), 5),
                "momentum": round(random.uniform(*self.search_space["momentum"]), 4),
                "weight_decay": round(random.uniform(*self.search_space["weight_decay"]), 6),
                "batch": random.choice(self.search_space["batch"]),
                "optimizer": random.choice(self.search_space["optimizer"]),
                "cos_lr": random.choice(self.search_space["cos_lr"]),
            }
            experiments.append(cfg)
        
        return experiments
    
    def load_or_create_experiments(self):
        """Load existing experiments or create new ones"""
        if os.path.exists(self.todo_json):
            with open(self.todo_json, "r") as f:
                todo = json.load(f)
            logger.info(f"Resuming from todo.json ({len(todo)} experiments remaining)")
        else:
            todo = self.generate_experiments()
            with open(self.todo_json, "w") as f:
                json.dump(todo, f, indent=2)
            logger.info(f"Created {len(todo)} experiments")
        
        return todo
    
    def setup_csv(self):
        """Setup CSV file for results"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "name", "lr0", "momentum", "weight_decay", "batch", 
                    "optimizer", "cos_lr", "mAP50", "mAP50_95", 
                    "precision", "recall", "status", "duration_min", "run_dir"
                ])
    
    def write_result(self, row):
        """Write result to CSV"""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    
    def run_experiment(self, cfg):
        """Run single experiment"""
        exp_name = cfg["name"]
        exp_dir = os.path.join(self.run_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        logger.info(f"{exp_name} | lr0={cfg['lr0']} momentum={cfg['momentum']} "
                   f"wd={cfg['weight_decay']} batch={cfg['batch']} "
                   f"opt={cfg['optimizer']} cosLR={cfg['cos_lr']}")
        
        start = time.time()
        status = "FAILED"
        map50 = map95 = mp = mr = 0.0
        
        try:
            model = YOLO(config.get('model', 'yolo11n.pt'))
            model.train(
                data=config.get('dataset_yaml', 'config.yaml'),
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=cfg["batch"],
                lr0=cfg["lr0"],
                momentum=cfg["momentum"],
                weight_decay=cfg["weight_decay"],
                optimizer=cfg["optimizer"],
                cos_lr=cfg["cos_lr"],
                amp=self.amp,
                device=self.device,
                workers=self.workers,
                patience=self.patience,
                save_period=self.save_period,
                project=self.run_dir,
                name=exp_name,
                pretrained=True,
                seed=42,
                verbose=False,
                exist_ok=True,
            )
            
            # Final validation
            val_results = model.val()
            
            map50 = float(val_results.box.map50)
            map95 = float(val_results.box.map)
            mp = float(val_results.box.mp)
            mr = float(val_results.box.mr)
            status = "SUCCESS"
            
        except Exception as e:
            logger.error(f"{exp_name} failed: {e}")
        
        duration = round((time.time() - start) / 60, 2)
        
        return [
            exp_name, cfg["lr0"], cfg["momentum"], cfg["weight_decay"], 
            cfg["batch"], cfg["optimizer"], cfg["cos_lr"], 
            map50, map95, mp, mr, status, duration, exp_dir
        ]
    
    def run(self):
        """Run hyperparameter search"""
        logger.info(f"\n{self.search_type.upper()} SEARCH FOR YOLOv11")
        logger.info(f"Output: {self.run_dir}")
        
        # Setup
        todo = self.load_or_create_experiments()
        self.setup_csv()
        
        logger.info(f"Total experiments: {len(todo)}\n")
        
        results = []
        
        while todo:
            cfg = todo[0]
            
            # Run experiment
            result = self.run_experiment(cfg)
            self.write_result(result)
            results.append((cfg["name"], result[7], result[8]))  # name, mAP50, mAP50-95
            
            # Remove completed experiment
            todo.pop(0)
            with open(self.todo_json, "w") as f:
                json.dump(todo, f, indent=2)
            
            # Clean GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Summary
        if results:
            results.sort(key=lambda x: x[1], reverse=True)
            logger.info("\nTop-3 configurations (by mAP50):")
            for rank, (name, m50, m95) in enumerate(results[:3], 1):
                logger.info(f"{rank}. {name}: mAP50={m50:.4f}, mAP50-95={m95:.4f}")
        
        # Create archive
        zip_path = self.run_dir + ".zip"
        shutil.make_archive(self.run_dir, "zip", self.run_dir)
        
        logger.info(f"\nResults CSV: {self.csv_path}")
        logger.info(f"Archive: {zip_path}")
        logger.info(f"Remaining experiments: {self.todo_json}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Hyperparameter Search')
    parser.add_argument('--type', type=str, default='random', 
                       choices=['random', 'grid'],
                       help='Search type: random or grid')
    parser.add_argument('--n', type=int, default=10,
                       help='Number of experiments (for random search)')
    
    args = parser.parse_args()
    
    search = HyperparameterSearch(
        search_type=args.type,
        n_experiments=args.n
    )
    search.run()


if __name__ == "__main__":
    main()