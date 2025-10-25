"""
Configuration management for Navigation Assist ML Pipeline
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Centralized configuration management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ValueError("Config file is empty")
            
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in config file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
    
    def _validate_config(self):
        """Validate required config fields"""
        required_fields = ['names', 'nc']
        missing = [field for field in required_fields if field not in self._config]
        
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")
        
        # Validate consistency
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"Mismatch: 'nc' is {self.num_classes} but 'names' has {len(self.class_names)} classes"
            )
    
    @property
    def class_names(self) -> list:
        """Get class names from config"""
        return self._config.get('names', [])
    
    @property
    def num_classes(self) -> int:
        """Get number of classes"""
        return self._config.get('nc', 0)
    
    @property
    def class_map(self) -> Dict[str, int]:
        """Get class name to ID mapping"""
        return {name.lower(): idx for idx, name in enumerate(self.class_names)}
    
    @property
    def train_path(self) -> str:
        """Get training dataset path"""
        return self._config.get('train', 'data/processed/train/images')
    
    @property
    def val_path(self) -> str:
        """Get validation dataset path"""
        return self._config.get('val', 'data/processed/val/images')
    
    @property
    def test_path(self) -> str:
        """Get test dataset path"""
        return self._config.get('test', 'data/processed/test/images')
    
    @property
    def model_path(self) -> str:
        """Get model path"""
        return self._config.get('model', 'yolo11n.pt')
    
    @property
    def dataset_yaml(self) -> str:
        """Get dataset YAML path"""
        return self._config.get('dataset_yaml', 'config.yaml')
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation)
        
        Examples:
            config.get('training.epochs')
            config.get('export.tflite.quantization')
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get all training parameters"""
        return self.get('training', {})
    
    def get_preprocessing_params(self) -> Dict[str, Any]:
        """Get all preprocessing parameters"""
        return self.get('preprocessing', {})
    
    def get_export_params(self) -> Dict[str, Any]:
        """Get all export parameters"""
        return self.get('export', {})
    
    def reload(self):
        """Reload configuration from file"""
        self._config = self._load_config()
        self._validate_config()
    
    def save(self, output_path: Optional[str] = None):
        """Save current configuration to file"""
        path = Path(output_path) if output_path else self.config_path
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    def update(self, key: str, value: Any):
        """
        Update configuration value (supports nested keys)
        
        Example:
            config.update('training.epochs', 200)
        """
        keys = key.split('.')
        current = self._config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def __repr__(self) -> str:
        return f"Config(path={self.config_path}, classes={self.num_classes})"
    
    def __str__(self) -> str:
        return f"Navigation Assist Config:\n" \
               f"  Classes: {self.class_names}\n" \
               f"  Model: {self.model_path}\n" \
               f"  Train: {self.train_path}"


# Global config instance
config = Config()


# Convenience functions
def get_config() -> Config:
    """Get global config instance"""
    return config


def reload_config():
    """Reload global config from file"""
    global config
    config.reload()