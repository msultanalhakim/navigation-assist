"""
Optimized Pipeline - Process in Memory, Save Only Final Results
Industrial best practice: minimize disk I/O and storage
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import random

from scripts.utils import logger, config, ensure_dir


@dataclass
class ImageData:
    """In-memory representation of image + label"""
    image_path: str
    label_path: str
    class_id: int
    class_name: str
    stem: str
    
    def __hash__(self):
        return hash(self.stem)


class DatasetCache:
    """Smart caching for dataset metadata"""
    
    def __init__(self, cache_dir: str = "cache/.dataset_cache"):
        self.cache_dir = Path(cache_dir)
        ensure_dir(self.cache_dir)
        
    def get_config_hash(self) -> str:
        """Generate hash from current config"""
        config_str = json.dumps(config._config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def is_valid(self) -> bool:
        """Check if cache is valid for current config"""
        cache_file = self.cache_dir / "config_hash.txt"
        if not cache_file.exists():
            return False
        
        cached_hash = cache_file.read_text().strip()
        current_hash = self.get_config_hash()
        return cached_hash == current_hash
    
    def save_hash(self):
        """Save current config hash"""
        cache_file = self.cache_dir / "config_hash.txt"
        cache_file.write_text(self.get_config_hash())
    
    def clear(self):
        """Clear cache"""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        ensure_dir(self.cache_dir)


class OptimizedDatasetProcessor:
    """Process dataset in memory, save only final results"""
    
    def __init__(self):
        self.cache = DatasetCache()
        self.class_map = config.class_map
        self.target_size = tuple(config.get('preprocessing', {}).get('resize', {}).get('target_size', [640, 640]))
        self.split_ratios = config.get('preprocessing', {}).get('split', {}).get('ratios', [0.7, 0.2, 0.1])
        self.seed = config.get('preprocessing', {}).get('split', {}).get('seed', 42)
        
    def discover_dataset(self) -> List[ImageData]:
        """
        Discover all valid image-label pairs from raw dataset
        Handles both single-class folders and mixed folders like dataset_primary
        """
        logger.info("Discovering dataset files...")
        
        raw_base = Path(config.get('preprocessing', {}).get('raw_path', 'data/raw'))
        discovered = []
        stats = {cls: 0 for cls in self.class_map}
        
        for folder in raw_base.iterdir():
            if not folder.is_dir() or folder.name.startswith('_'):
                continue
            
            folder_lc = folder.name.lower()
            
            # Detect structure: images/labels subfolders or flat
            images_dir = folder / "images" if (folder / "images").exists() else folder
            labels_dir = folder / "labels" if (folder / "labels").exists() else folder
            
            # Find all label files
            for label_file in labels_dir.glob("*.txt"):
                if label_file.stat().st_size == 0:
                    continue
                
                # Read label to determine class
                # For mixed folders, we detect class from label content
                with open(label_file, 'r') as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue
                    
                    # Get class_id from label file
                    original_class_id = int(first_line.split()[0])
                
                # Determine class based on folder name OR label content
                if folder_lc in self.class_map:
                    # Single-class folder (person, chair, etc.)
                    class_id = self.class_map[folder_lc]
                    class_name = folder_lc
                else:
                    # Mixed folder (dataset_primary) - use class from label
                    # Map original class_id to our class_names
                    if original_class_id < len(config.class_names):
                        class_name = config.class_names[original_class_id]
                        class_id = original_class_id
                    else:
                        logger.warning(f"Unknown class_id {original_class_id} in {label_file.name}")
                        continue
                
                # Find corresponding image
                stem = label_file.stem
                image_path = None
                
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    img_candidate = images_dir / f"{stem}{ext}"
                    if img_candidate.exists():
                        image_path = str(img_candidate)
                        break
                
                if image_path:
                    discovered.append(ImageData(
                        image_path=image_path,
                        label_path=str(label_file),
                        class_id=class_id,
                        class_name=class_name,
                        stem=f"{folder.name}_{stem}"  # Add folder prefix to avoid name collision
                    ))
                    stats[class_name] = stats.get(class_name, 0) + 1
        
        logger.info(f"Discovered {len(discovered)} valid image-label pairs")
        for cls, count in sorted(stats.items()):
            if count > 0:
                logger.info(f"  {cls}: {count}")
        
        return discovered
    
    def process_and_save_image(self, args: Tuple[ImageData, Path, int]) -> bool:
        """Process single image: resize, remap labels, save to final location"""
        img_data, output_base, new_class_id = args
        
        try:
            # Read and resize image
            with Image.open(img_data.image_path) as img:
                img = img.convert('RGB')
                img = img.resize(self.target_size, Image.BILINEAR)
                
                # Save to final location
                output_img = output_base / f"{img_data.stem}.jpg"
                img.save(output_img, quality=95, format='JPEG')
            
            # Read, remap, and save label
            with open(img_data.label_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    parts[0] = str(new_class_id)  # Remap class ID
                    new_lines.append(" ".join(parts) + "\n")
            
            output_label = output_base.parent / "labels" / f"{img_data.stem}.txt"
            with open(output_label, 'w') as f:
                f.writelines(new_lines)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {img_data.stem}: {e}")
            return False
    
    def split_and_save(self, dataset: List[ImageData]):
        """
        Split dataset and save directly to final locations
        No intermediate files created
        """
        logger.info("Splitting and saving dataset...")
        
        output_base = Path(config.get('preprocessing', {}).get('split', {}).get('output_path', 'data/processed'))
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            ensure_dir(output_base / split / 'images')
            ensure_dir(output_base / split / 'labels')
        
        # Shuffle dataset
        random.seed(self.seed)
        random.shuffle(dataset)
        
        # Calculate split indices
        n = len(dataset)
        train_end = int(self.split_ratios[0] * n)
        val_end = train_end + int(self.split_ratios[1] * n)
        
        splits = {
            'train': dataset[:train_end],
            'val': dataset[train_end:val_end],
            'test': dataset[val_end:]
        }
        
        # Process and save in parallel
        total_processed = 0
        
        for split_name, split_data in splits.items():
            logger.info(f"Processing {split_name} split: {len(split_data)} images")
            
            split_base = output_base / split_name / 'images'
            
            # Prepare args for parallel processing
            process_args = [
                (img_data, split_base, img_data.class_id)
                for img_data in split_data
            ]
            
            # Process in parallel
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(self.process_and_save_image, args) for args in process_args]
                
                for future in as_completed(futures):
                    if future.result():
                        total_processed += 1
        
        logger.info(f"Successfully processed and saved {total_processed} images")
        
        # Save split metadata
        metadata = {
            'total': len(dataset),
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test']),
            'class_map': self.class_map,
            'target_size': self.target_size,
            'split_ratios': self.split_ratios
        }
        
        with open(output_base / 'dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return splits
    
    def run(self, force: bool = False):
        """
        Run optimized pipeline
        
        Args:
            force: Force reprocessing even if cache is valid
        """
        logger.info("=" * 80)
        logger.info("OPTIMIZED DATASET PROCESSING PIPELINE")
        logger.info("=" * 80)
        
        # Check cache
        if not force and self.cache.is_valid():
            logger.info("Valid cache found. Use --force to reprocess.")
            logger.info("Skipping dataset processing...")
            return
        
        # Clear old cache
        self.cache.clear()
        
        # Step 1: Discover all valid pairs (in memory)
        dataset = self.discover_dataset()
        
        if not dataset:
            logger.error("No valid image-label pairs found!")
            return
        
        # Step 2: Process and save directly to final location
        # This combines: merge, remap, resize, and split in one pass
        splits = self.split_and_save(dataset)
        
        # Step 3: Verify
        logger.info("\nVerifying processed dataset...")
        output_base = Path(config.get('preprocessing', {}).get('split', {}).get('output_path', 'data/processed'))
        
        for split_name in ['train', 'val', 'test']:
            img_count = len(list((output_base / split_name / 'images').glob('*.jpg')))
            lbl_count = len(list((output_base / split_name / 'labels').glob('*.txt')))
            logger.info(f"{split_name}: {img_count} images, {lbl_count} labels")
        
        # Save cache
        self.cache.save_hash()
        
        logger.info("\n" + "=" * 80)
        logger.info("DATASET PROCESSING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Output: {output_base}")
        logger.info("NOTE: No intermediate files created - processed in memory")


def main(force: bool = False):
    """Main entry point"""
    processor = OptimizedDatasetProcessor()
    processor.run(force=force)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Optimized Dataset Processing')
    parser.add_argument('--force', action='store_true', help='Force reprocessing')
    args = parser.parse_args()
    
    main(force=args.force)