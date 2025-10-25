#!/usr/bin/env python3
"""
Verify dataset annotations - Check if labels match images
"""

from pathlib import Path
import json
from collections import defaultdict, Counter

# ============================================================================
# CONFIG
# ============================================================================

DATASET_ROOT = Path("dataset") / "processed"
CLASS_NAMES = ["person", "chair", "table", "door", "stair"]

# Files to investigate based on test results
SUSPICIOUS_FILES = [
    "10-129.jpg",              # Expected: chair, Got: person
    "65_dup1.jpg",             # Expected: chair, Got: person
    "cadeira-de-escritorio-azul_mp4-56.jpg",  # Expected: chair, Got: person
    "door-7.jpg",              # Expected: door, Got: chair
    "door-76.jpg",             # Expected: door, Got: chair
    "WhatsApp-Image-2025-09-29-at-17_28_23-1-_jpeg.rf.11e35fb28317800c5b0a3bc19bf6fbe1.jpg",  # Expected: table, Got: stair
]

# ============================================================================
# FUNCTIONS
# ============================================================================

def find_file_in_dataset(filename: str):
    """Find a file across all dataset splits"""
    for split in ['train', 'val', 'test']:
        img_path = DATASET_ROOT / split / 'images' / filename
        label_path = DATASET_ROOT / split / 'labels' / (Path(filename).stem + '.txt')
        
        if img_path.exists():
            return split, img_path, label_path
    
    return None, None, None


def parse_label_file(label_path: Path):
    """Parse YOLO format label file"""
    if not label_path.exists():
        return []
    
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                annotations.append({
                    'class_id': class_id,
                    'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"UNKNOWN_{class_id}",
                    'bbox': [x_center, y_center, width, height]
                })
    
    return annotations


def count_dataset_distribution():
    """Count class distribution across dataset"""
    distribution = {
        'train': Counter(),
        'val': Counter(),
        'test': Counter()
    }
    
    for split in ['train', 'val', 'test']:
        label_dir = DATASET_ROOT / split / 'labels'
        if not label_dir.exists():
            continue
        
        for label_file in label_dir.glob('*.txt'):
            annotations = parse_label_file(label_file)
            for ann in annotations:
                distribution[split][ann['class_name']] += 1
    
    return distribution


def analyze_filename_patterns():
    """Analyze if filenames suggest their content"""
    patterns = defaultdict(list)
    
    for split in ['train', 'val', 'test']:
        label_dir = DATASET_ROOT / split / 'labels'
        if not label_dir.exists():
            continue
        
        for label_file in label_dir.glob('*.txt'):
            annotations = parse_label_file(label_file)
            if annotations:
                filename = label_file.stem
                classes_in_file = set([ann['class_name'] for ann in annotations])
                
                # Check if filename contains class hints
                filename_lower = filename.lower()
                for class_name in CLASS_NAMES:
                    if class_name in filename_lower:
                        patterns[class_name].append({
                            'filename': filename,
                            'expected': class_name,
                            'actual': list(classes_in_file),
                            'match': class_name in classes_in_file
                        })
    
    return patterns


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("DATASET ANNOTATION VERIFICATION")
    print("=" * 80)
    
    # Check if dataset exists
    if not DATASET_ROOT.exists():
        print(f"❌ ERROR: Dataset not found at {DATASET_ROOT}")
        return
    
    print(f"\n📁 Dataset root: {DATASET_ROOT}\n")
    
    # ========================================================================
    # 1. INVESTIGATE SUSPICIOUS FILES
    # ========================================================================
    
    print("=" * 80)
    print("1️⃣  INVESTIGATING SUSPICIOUS FILES")
    print("=" * 80 + "\n")
    
    issues_found = []
    
    for filename in SUSPICIOUS_FILES:
        print(f"📄 {filename}")
        
        split, img_path, label_path = find_file_in_dataset(filename)
        
        if split is None:
            print(f"  ❌ File not found in dataset!\n")
            continue
        
        print(f"  📍 Location: {split} split")
        
        if label_path.exists():
            annotations = parse_label_file(label_path)
            
            if annotations:
                print(f"  📝 Annotations in label file:")
                class_counts = Counter([ann['class_name'] for ann in annotations])
                for class_name, count in class_counts.most_common():
                    print(f"     - {class_name}: {count} object(s)")
                
                # Check filename vs content
                filename_lower = filename.lower()
                for class_name in CLASS_NAMES:
                    if class_name in filename_lower:
                        if class_name not in class_counts:
                            print(f"  ⚠️  MISMATCH: Filename suggests '{class_name}' but label has: {list(class_counts.keys())}")
                            issues_found.append({
                                'file': filename,
                                'expected_from_name': class_name,
                                'actual_label': list(class_counts.keys())
                            })
            else:
                print(f"  ⚠️  Label file is EMPTY!")
        else:
            print(f"  ❌ Label file NOT FOUND: {label_path.name}")
        
        print()
    
    # ========================================================================
    # 2. DATASET DISTRIBUTION
    # ========================================================================
    
    print("=" * 80)
    print("2️⃣  DATASET CLASS DISTRIBUTION")
    print("=" * 80 + "\n")
    
    distribution = count_dataset_distribution()
    
    for split in ['train', 'val', 'test']:
        print(f"{split.upper()} split:")
        if distribution[split]:
            total = sum(distribution[split].values())
            for class_name in CLASS_NAMES:
                count = distribution[split][class_name]
                percent = (count / total * 100) if total > 0 else 0
                print(f"  {class_name:10s}: {count:5d} ({percent:5.1f}%)")
        else:
            print(f"  No data found")
        print()
    
    # ========================================================================
    # 3. FILENAME PATTERN ANALYSIS
    # ========================================================================
    
    print("=" * 80)
    print("3️⃣  FILENAME vs LABEL CONSISTENCY CHECK")
    print("=" * 80 + "\n")
    
    patterns = analyze_filename_patterns()
    
    mismatches = 0
    for class_name in CLASS_NAMES:
        if class_name in patterns:
            files_with_class_in_name = patterns[class_name]
            matches = sum(1 for f in files_with_class_in_name if f['match'])
            total = len(files_with_class_in_name)
            
            print(f"Files with '{class_name}' in filename: {total}")
            print(f"  ✓ Correctly labeled: {matches}")
            print(f"  ❌ Mislabeled: {total - matches}")
            
            if total - matches > 0:
                print(f"  Examples of mismatches:")
                for f in files_with_class_in_name[:3]:
                    if not f['match']:
                        print(f"    - {f['filename']}: expected '{f['expected']}', got {f['actual']}")
                        mismatches += 1
            print()
    
    # ========================================================================
    # 4. SUMMARY & RECOMMENDATIONS
    # ========================================================================
    
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80 + "\n")
    
    print(f"Suspicious files investigated: {len(SUSPICIOUS_FILES)}")
    print(f"Issues found: {len(issues_found)}")
    print(f"Filename-label mismatches: {mismatches}")
    
    if len(issues_found) > 0 or mismatches > 0:
        print("\n" + "=" * 80)
        print("⚠️  CRITICAL ISSUES DETECTED!")
        print("=" * 80)
        print("\nYour dataset has labeling problems. This explains why:")
        print("  - Chairs are detected as persons")
        print("  - Doors are detected as chairs")
        print("  - Tables are detected as stairs")
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("  1. Re-check your annotation files (.txt in labels/ folders)")
        print("  2. Verify class IDs match: 0=person, 1=chair, 2=table, 3=door, 4=stair")
        print("  3. Use labelImg or similar tool to visually verify annotations")
        print("  4. Consider re-annotating the dataset or fixing the label files")
        print("  5. Retrain the model after fixing annotations")
    else:
        print("\n✓ No obvious issues found in annotations")
        print("  The model confusion might be due to:")
        print("  - Similar visual features between classes")
        print("  - Insufficient training data")
        print("  - Need for more training epochs")
    
    print("\n" + "=" * 80)
    
    # Save detailed report
    report = {
        'suspicious_files': issues_found,
        'distribution': {
            split: dict(distribution[split])
            for split in ['train', 'val', 'test']
        },
        'filename_patterns': {
            class_name: [
                {k: v for k, v in f.items() if k != 'bbox'}
                for f in patterns[class_name][:10]  # First 10 examples
            ]
            for class_name in patterns
        }
    }
    
    with open('annotation_verification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📄 Detailed report saved to: annotation_verification_report.json")
    print("=" * 80)


if __name__ == "__main__":
    main()