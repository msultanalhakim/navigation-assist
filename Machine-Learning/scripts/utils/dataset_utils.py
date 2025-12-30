from collections import Counter
from pathlib import Path

def count_classes(labels_dir: str, class_names: list[str]) -> dict:
    """Hitung distribusi class di folder labels."""
    counter = Counter()
    labels_path = Path(labels_dir)

    for file in labels_path.glob("*.txt"):
        with open(file, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                cls_name = class_names[cls]
                counter[cls_name] += 1

    return dict(counter)


def label_exists(label_path: str) -> bool:
    """Cek apakah label file ada dan tidak kosong."""
    p = Path(label_path)
    return p.exists() and p.stat().st_size > 0
