# Resizing Gambar ke 640×640
import os
from PIL import Image
from tqdm.auto import tqdm
import shutil

# --- path sumber & tujuan ---
base_path     = os.path.join("dataset", "raw", "_merged")
images_path   = os.path.join(base_path, "images")
labels_path   = os.path.join(base_path, "labels")

out_base      = os.path.join("dataset", "raw", "_merged_resized")
out_images    = os.path.join(out_base, "images")
out_labels    = os.path.join(out_base, "labels")
os.makedirs(out_images, exist_ok=True)
os.makedirs(out_labels, exist_ok=True)

# --- parameter resize ---
target_size = (640, 640)

# --- proses semua gambar yang ada pasangannya (label) ---
files = [f for f in os.listdir(images_path) if f.lower().endswith((".jpg",".jpeg",".png"))]
processed = 0
skipped_no_label = 0

for fname in tqdm(files, desc="Resizing"):
    stem, ext = os.path.splitext(fname)
    src_img   = os.path.join(images_path, fname)
    src_lbl   = os.path.join(labels_path, stem + ".txt")

    # hanya proses jika label ada & tidak kosong
    if not os.path.exists(src_lbl) or os.path.getsize(src_lbl) == 0:
        skipped_no_label += 1
        continue

    try:
        with Image.open(src_img) as im:
            im = im.convert("RGB")                 # normalisasi ke RGB
            im = im.resize(target_size, Image.BILINEAR)
            im.save(os.path.join(out_images, stem + ".jpg"), quality=95)  # simpan sebagai .jpg
        # salin label apa adanya (YOLO normalized)
        shutil.copy(src_lbl, os.path.join(out_labels, stem + ".txt"))
        processed += 1
    except Exception as e:
        tqdm.write(f"Gagal {fname}: {e}")

print(f"\nSelesai: {processed} gambar di-resize ke {target_size[0]}×{target_size[1]}")
print(f"Dilewati karena tidak ada/label kosong: {skipped_no_label}")
