# 📘 4.2.4 Image Augmentation Otomatis ke Seluruh Dataset Training (Versi Aman)
import os
import cv2
import albumentations as A

# === Path input dan output ===
base_input = os.path.join("dataset", "processed", "train")
images_dir = os.path.join(base_input, "images")
labels_dir = os.path.join(base_input, "labels")

base_output = os.path.join("dataset", "processed", "train_augmented")
output_images = os.path.join(base_output, "images")
output_labels = os.path.join(base_output, "labels")
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# === Definisi transformasi augmentasi ===
transform = A.Compose(
    [
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1),
                 rotate=0, shear=0, cval=(114, 114, 114), p=0.5),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        clip=True,                      # <== biar otomatis dipotong ke [0,1]
        min_visibility=0.2,
    ),
)

# === Fungsi bantu ===
def load_yolo_label(label_path):
    """Membaca label YOLO (class x_center y_center width height)."""
    boxes, classes = [], []
    if not os.path.exists(label_path):
        return boxes, classes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = map(float, parts)
            boxes.append([x, y, w, h])
            classes.append(int(cls))
    return boxes, classes


def save_yolo_label(label_path, boxes, classes):
    """Menyimpan label YOLO ke file."""
    with open(label_path, "w") as f:
        for cls, (x, y, w, h) in zip(classes, boxes):
            # pastikan tetap di [0,1]
            x, y, w, h = [max(0, min(1, v)) for v in [x, y, w, h]]
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


# === Proses augmentasi ===
count, skipped = 0, 0
for filename in os.listdir(images_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    name = os.path.splitext(filename)[0]
    img_path = os.path.join(images_dir, filename)
    lbl_path = os.path.join(labels_dir, f"{name}.txt")

    # Muat gambar dan label
    image = cv2.imread(img_path)
    if image is None:
        skipped += 1
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, classes = load_yolo_label(lbl_path)
    if not boxes:
        skipped += 1
        continue

    try:
        # Terapkan augmentasi
        augmented = transform(image=image, bboxes=boxes, class_labels=classes)

        aug_img = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
        aug_boxes = augmented["bboxes"]
        aug_classes = augmented["class_labels"]

        # Simpan hasil augmentasi
        new_name = f"aug_{name}.jpg"
        cv2.imwrite(os.path.join(output_images, new_name), aug_img)
        save_yolo_label(os.path.join(output_labels, f"aug_{name}.txt"), aug_boxes, aug_classes)

        count += 1
        if count % 100 == 0:
            print(f"{count} gambar sudah diaugmentasi...")

    except Exception as e:
        skipped += 1
        print(f"Gagal augmentasi {filename}: {e}")

# === Ringkasan hasil ===
print(f"\nAugmentasi selesai!")
print(f"Total berhasil: {count} gambar")
print(f"Total dilewati: {skipped} gambar (label kosong / error)")
print(f"📂 Output: {output_images}")
