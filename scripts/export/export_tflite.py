"""
Export YOLOv11 model to TFLite format
"""
import os
from pathlib import Path
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils import logger, config


class TFLiteExporter:
    """Handle model export to TFLite"""
    
    def __init__(self, model_path: str = None):
        # Get model path
        if model_path is None:
            project = config.get('project', 'runs/train')
            experiment = config.get('experiment', 'exp_latest')
            model_path = f"{project}/{experiment}/weights/best.pt"
        
        self.model_path = model_path
        self.output_path = Path(config.get('export', {}).get('output_path', 'models'))
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load model
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
    
    def export_float32(self):
        """Export to TFLite Float32"""
        logger.info("Exporting to TFLite Float32...")
        
        try:
            output = self.model.export(
                format='tflite',
                imgsz=640,
                int8=False,
                half=False
            )
            
            # Move to models folder with clear name
            if output:
                src = Path(output)
                dst = self.output_path / "best_float32.tflite"
                if src.exists():
                    src.rename(dst)
                    logger.info(f"Float32 model saved: {dst}")
                    return str(dst)
        except Exception as e:
            logger.error(f"Float32 export failed: {e}")
        
        return None
    
    def export_float16(self):
        """Export to TFLite Float16 (smaller size)"""
        logger.info("Exporting to TFLite Float16...")
        
        try:
            output = self.model.export(
                format='tflite',
                imgsz=640,
                int8=False,
                half=True
            )
            
            if output:
                src = Path(output)
                dst = self.output_path / "best_float16.tflite"
                if src.exists():
                    src.rename(dst)
                    logger.info(f"Float16 model saved: {dst}")
                    return str(dst)
        except Exception as e:
            logger.error(f"Float16 export failed: {e}")
        
        return None
    
    def export_int8(self, calibration_images: int = 100):
        """Export to TFLite INT8 (quantized, smallest size)"""
        logger.info(f"Exporting to TFLite INT8 (using {calibration_images} calibration images)...")
        
        try:
            output = self.model.export(
                format='tflite',
                imgsz=640,
                int8=True,
                data=config.get('dataset_yaml', 'config.yaml')
            )
            
            if output:
                src = Path(output)
                dst = self.output_path / "best_int8.tflite"
                if src.exists():
                    src.rename(dst)
                    logger.info(f"INT8 model saved: {dst}")
                    return str(dst)
        except Exception as e:
            logger.error(f"INT8 export failed: {e}")
        
        return None
    
    def export_all(self):
        """Export all TFLite variants"""
        logger.info("=" * 80)
        logger.info("EXPORTING MODEL TO TFLITE")
        logger.info("=" * 80)
        logger.info(f"Source model: {self.model_path}")
        logger.info(f"Output directory: {self.output_path}\n")
        
        results = {}
        
        # Get quantization settings from config
        quantization = config.get('export', {}).get('tflite', {}).get('quantization', ['float16', 'int8'])
        
        if 'float32' in quantization:
            results['float32'] = self.export_float32()
        
        if 'float16' in quantization:
            results['float16'] = self.export_float16()
        
        if 'int8' in quantization:
            cal_images = config.get('export', {}).get('tflite', {}).get('int8_calibration_images', 100)
            results['int8'] = self.export_int8(cal_images)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("EXPORT SUMMARY")
        logger.info("=" * 80)
        
        for variant, path in results.items():
            if path and Path(path).exists():
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                logger.info(f"{variant.upper():10s}: {path} ({size_mb:.2f} MB)")
            else:
                logger.info(f"{variant.upper():10s}: FAILED")
        
        logger.info("=" * 80)
        
        return results


def main(model_path: str = None):
    """Main export function"""
    exporter = TFLiteExporter(model_path)
    exporter.export_all()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Export YOLOv11 to TFLite')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (.pt)')
    
    args = parser.parse_args()
    main(args.model)