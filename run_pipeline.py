"""
NAVIGATION ASSIST ML PIPELINE (OPTIMIZED)
------------------------------------
Single-pass preprocessing -> training -> export
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.preprocessing.optimized_pipeline import main as preprocessing_step
from scripts.preprocessing.verify_dataset_annotations import main as verify_step
from scripts.training.train_yolo import main as train_step
from scripts.export.export_tflite import main as export_step

from scripts.utils import logger, config


def main():
    """Execute optimized ML pipeline"""
    
    logger.info("=" * 80)
    logger.info("NAVIGATION ASSIST PIPELINE (OPTIMIZED)")
    logger.info("=" * 80)
    logger.info(f"Configuration: {config.config_path}")
    logger.info(f"Classes: {config.class_names}")
    logger.info(f"Number of classes: {config.num_classes}")
    logger.info("=" * 80)
    
    try:
        # Step 1: Preprocessing (Single-pass: merge, remap, resize, split)
        logger.info("\nStep 1: Dataset Processing (Optimized Pipeline)")
        logger.info("-" * 80)
        preprocessing_step()
        
        # Step 2: Verify Annotations
        logger.info("\nStep 2: Verify Annotations")
        logger.info("-" * 80)
        verify_step()
        
        # Step 3: Train YOLOv11 (with built-in augmentation)
        logger.info("\nStep 3: Train YOLOv11")
        logger.info("-" * 80)
        logger.info("Note: Using YOLO built-in augmentation (real-time)")
        train_step()
        
        # Step 4: Export TFLite (optional)
        if 'tflite' in config.get('export', {}).get('formats', []):
            logger.info("\nStep 4: Export TFLite")
            logger.info("-" * 80)
            export_step()
        else:
            logger.info("\nStep 4: Export - SKIPPED (tflite not in export formats)")
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\nPipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()