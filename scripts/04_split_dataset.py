# Split Dataset ke Train/Val/Test
import os, random, shutil

# Path dataset hasil resize
base_path = os.path.join("dataset", "raw", "_merged_resized")
images_dir = os.path.join(base_path, "images")
labels_dir = os.path.join(base_path, "labels")

# Folder output
output_base = os.path.join("dataset", "processed")
splits = ["train", "val", "test"]
ratios = [0.7, 0.2, 0.1]

for s in splits:
    os.makedirs(os.path.join(output_base, s, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base, s, "labels"), exist_ok=True)

# Ambil semua nama file tanpa ekstensi
files = [f[:-4] for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]
random.shuffle(files)

n = len(files)
train_end = int(ratios[0]*n)
val_end = train_end + int(ratios[1]*n)

splits_dict = {
    "train": files[:train_end],
    "val": files[train_end:val_end],
    "test": files[val_end:]
}

for split, names in splits_dict.items():
    for name in names:
        img_src = os.path.join(images_dir, f"{name}.jpg")
        lbl_src = os.path.join(labels_dir, f"{name}.txt")
        img_dst = os.path.join(output_base, split, "images", f"{name}.jpg")
        lbl_dst = os.path.join(output_base, split, "labels", f"{name}.txt")

        shutil.copy(img_src, img_dst)
        shutil.copy(lbl_src, lbl_dst)

print("Dataset dibagi menjadi train/val/test di folder 'dataset/processed'")
