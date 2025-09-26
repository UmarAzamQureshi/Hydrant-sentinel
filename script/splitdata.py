import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

"""
This script merges images from both Hydrant and Car folders into a single
YOLOv8 dataset at `hydrant_dataset/` while preserving correct label files.

Assumptions:
- Hydrant labels are in `labels/train` (class 0) created by auto_label_hydrants.py
- Car labels are in `labels/car` (class 1) created by auto_label_car.py
"""

# ---------------- CONFIG ----------------
# Original images and labels
HYDRANT_IMAGE_FOLDER = r"D:\Hydrant-sentinel\Hydrant-sentinel\Hydrant"
HYDRANT_LABEL_FOLDER = r"D:\Hydrant-sentinel\Hydrant-sentinel\labels\train"
CAR_IMAGE_FOLDER     = r"D:\Hydrant-sentinel\Hydrant-sentinel\Car"
CAR_LABEL_FOLDER     = r"D:\Hydrant-sentinel\Hydrant-sentinel\labels\car"

# Destination folders (YOLOv8 structure)
IMAGE_TRAIN = r"D:\Hydrant-sentinel\Hydrant-sentinel\hydrant_dataset\images\train"
IMAGE_VAL   = r"D:\Hydrant-sentinel\Hydrant-sentinel\hydrant_dataset\images\val"
LABEL_TRAIN = r"D:\Hydrant-sentinel\Hydrant-sentinel\hydrant_dataset\labels\train"
LABEL_VAL   = r"D:\Hydrant-sentinel\Hydrant-sentinel\hydrant_dataset\labels\val"

# ---------------- CREATE DESTINATION FOLDERS ----------------
for folder in [IMAGE_TRAIN, IMAGE_VAL, LABEL_TRAIN, LABEL_VAL]:
    os.makedirs(folder, exist_ok=True)

# ---------------- GET ALL IMAGE FILES ----------------
hydrant_images = [f for f in os.listdir(HYDRANT_IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
car_images = [f for f in os.listdir(CAR_IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Prefix to avoid filename collisions when copying into one dataset
hydrant_prefixed = [("hydrant__" + f, f, "hydrant") for f in hydrant_images]
car_prefixed     = [("car__" + f, f, "car") for f in car_images]

all_items = hydrant_prefixed + car_prefixed
all_filenames = [dst for dst, _, _ in all_items]

# ---------------- SPLIT DATA ----------------
train_items, val_items = train_test_split(all_items, test_size=0.2, random_state=42)

# ---------------- COPY TRAIN FILES ----------------
for dst_name, src_name, cls in train_items:
    if cls == "hydrant":
        src_img_dir = HYDRANT_IMAGE_FOLDER
        src_lbl_dir = HYDRANT_LABEL_FOLDER
    else:
        src_img_dir = CAR_IMAGE_FOLDER
        src_lbl_dir = CAR_LABEL_FOLDER

    # Copy image (with prefixed name)
    shutil.copy(os.path.join(src_img_dir, src_name), os.path.join(IMAGE_TRAIN, dst_name))

    # Copy label if exists (rename to match prefixed image stem)
    label_file = os.path.splitext(src_name)[0] + ".txt"
    label_src_path = os.path.join(src_lbl_dir, label_file)
    if os.path.exists(label_src_path):
        label_dst_path = os.path.join(LABEL_TRAIN, os.path.splitext(dst_name)[0] + ".txt")
        shutil.copy(label_src_path, label_dst_path)

# ---------------- COPY VAL FILES ----------------
for dst_name, src_name, cls in val_items:
    if cls == "hydrant":
        src_img_dir = HYDRANT_IMAGE_FOLDER
        src_lbl_dir = HYDRANT_LABEL_FOLDER
    else:
        src_img_dir = CAR_IMAGE_FOLDER
        src_lbl_dir = CAR_LABEL_FOLDER

    shutil.copy(os.path.join(src_img_dir, src_name), os.path.join(IMAGE_VAL, dst_name))

    label_file = os.path.splitext(src_name)[0] + ".txt"
    label_src_path = os.path.join(src_lbl_dir, label_file)
    if os.path.exists(label_src_path):
        label_dst_path = os.path.join(LABEL_VAL, os.path.splitext(dst_name)[0] + ".txt")
        shutil.copy(label_src_path, label_dst_path)

print(f"Dataset split complete!")
print(f"Train images: {len(train_items)}, Val images: {len(val_items)}")
