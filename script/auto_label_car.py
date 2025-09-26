from ultralytics import YOLO
import os
from pathlib import Path

# ---------------- CONFIG ----------------
IMAGE_FOLDER = "D:/Hydrant-sentinel/Hydrant-sentinel/Car"
LABEL_FOLDER = "D:/Hydrant-sentinel/Hydrant-sentinel/labels/car"

MODEL = "yolov8n.pt"                             # Pre-trained YOLOv8 model
CONF_THRESHOLD = 0.3                              # Confidence threshold for predictions

# Make sure label folder exists
os.makedirs(LABEL_FOLDER, exist_ok=True)

# Load YOLOv8 pre-trained model
model = YOLO(MODEL)

# Loop over all images
for img_name in os.listdir(IMAGE_FOLDER):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMAGE_FOLDER, img_name)
    results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)

    # Save in YOLO format
    label_path = os.path.join(LABEL_FOLDER, Path(img_name).stem + ".txt")
    with open(label_path, "w") as f:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = box.conf[0].item() if hasattr(box, "conf") else 1.0

            # Skip low-confidence predictions
            if score < CONF_THRESHOLD:
                continue

            # Convert to YOLO format (normalized)
            img_h, img_w = results[0].orig_shape
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            # class_id = 1 (car)
            f.write(f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"Labeled: {img_name}")
    results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
    print(f"{img_name}: {len(results[0].boxes)} boxes detected")


