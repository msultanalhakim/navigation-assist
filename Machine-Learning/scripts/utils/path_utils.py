import os
from pathlib import Path

def ensure_dir(path: str | Path) -> None:
    """Membuat folder jika belum ada."""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_stem(filename: str) -> str:
    """Ambil nama file tanpa ekstensi."""
    return Path(filename).stem

def list_files(directory: str, ext: tuple[str] = (".jpg", ".jpeg", ".png")):
    """List file dalam folder, filter by extension."""
    return [
        f for f in os.listdir(directory)
        if f.lower().endswith(ext)
    ]
