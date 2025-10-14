import os
import re
import shutil

# === Base path dataset ===
base_path = os.path.join("dataset", "raw")

# === Mapping class_id berdasarkan nama folder (pakai lowercase untuk konsistensi) ===
class_map = {
    "person": 0,
    "chair": 1,
    "table": 2,
    "door": 3,
    "stair-case": 4,
}

# === Folder output gabungan ===
output_images = os.path.join(base_path, "_merged", "images")
output_labels = os.path.join(base_path, "_merged", "labels")
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# === Fungsi pembuat nama unik ===
def unique_name(dst_dir: str, filename: str) -> str:
    """Jika filename sudah ada di dst_dir, tambahkan _dupX sebelum ekstensi."""
    name, ext = os.path.splitext(filename)
    cand = filename
    i = 1
    while os.path.exists(os.path.join(dst_dir, cand)):
        cand = f"{name}_dup{i}{ext}"
        i += 1
    return cand

# === Fungsi pembersih nama ===
_pat_rf = re.compile(r"\.rf\.[a-f0-9]{10,}", re.IGNORECASE)
_pat_exttag = re.compile(r"([_\-](jpeg|jpg|png))+$", re.IGNORECASE)
_pat_multius = re.compile(r"[_\-]{2,}")

def clean_stem(stem: str) -> str:
    s = _pat_rf.sub("", stem)              # hapus .rf.<hash>
    s = _pat_exttag.sub("", s)             # hapus _jpg / -jpg / _png / -png / _jpeg
    s = s.strip("._- ")                    # bersihkan karakter tepi
    s = _pat_multius.sub("_", s)           # rapikan separator berulang
    return s.lower()

# === Counters ===
class_counts = {cls: 0 for cls in class_map}
empty_counts = {cls: 0 for cls in class_map}
total_processed = 0
total_empty = 0

# === Loop setiap folder kelas di dataset/raw ===
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)

    # Skip helper folders seperti _merged, _combined, dsb
    if not os.path.isdir(folder_path) or folder.startswith("_"):
        continue

    folder_lc = folder.lower()
    class_id = class_map.get(folder_lc)
    if class_id is None:
        print(f"Folder {folder_lc} tidak ada di class_map, dilewati.")
        continue

    # Deteksi otomatis: ada subfolder images/labels atau campur
    images_dir = os.path.join(folder_path, "images")
    labels_dir = os.path.join(folder_path, "labels")
    if not (os.path.isdir(images_dir) and os.path.isdir(labels_dir)):
        images_dir = folder_path
        labels_dir = folder_path

    processed_count = 0
    empty_count = 0

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(labels_dir, label_file)

        # Lewati label kosong
        if os.path.getsize(label_path) == 0:
            empty_count += 1
            continue

        # Baca & remap class id
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            parts[0] = str(class_id)
            new_lines.append(" ".join(parts) + "\n")

        # Ambil gambar pasangan
        stem = os.path.splitext(label_file)[0]
        img_src = None
        for ext in (".jpg", ".jpeg", ".png"):
            path = os.path.join(images_dir, stem + ext)
            if os.path.exists(path):
                img_src = path
                break

        if not img_src:
            os.remove(os.path.join(output_labels, label_file)) if os.path.exists(label_file) else None
            continue

        # === Bersihkan nama file ===
        raw_stem = os.path.splitext(os.path.basename(img_src))[0]
        clean = clean_stem(raw_stem)
        img_ext = os.path.splitext(img_src)[1].lower()

        # Simpan nama unik
        dst_img_name = unique_name(output_images, f"{clean}{img_ext}")
        dst_lbl_name = os.path.splitext(dst_img_name)[0] + ".txt"

        # Simpan file label & gambar
        with open(os.path.join(output_labels, dst_lbl_name), "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        shutil.copy(img_src, os.path.join(output_images, dst_img_name))
        processed_count += 1

    class_counts[folder_lc] = processed_count
    empty_counts[folder_lc] = empty_count
    total_processed += processed_count
    total_empty += empty_count

    print(f"📂 {folder_lc}: {processed_count} file valid, {empty_count} label kosong di-skip.")

# === Rekapitulasi ===
if total_processed > 0:
    print(f"\nSemua dataset berhasil digabung ({total_processed} file).")
    print(f"Total label kosong di-skip: {total_empty}")
else:
    print("\nTidak ada data valid yang berhasil digabung.")
