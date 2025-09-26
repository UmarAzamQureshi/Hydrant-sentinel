import os
import math
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
from ultralytics import YOLO


# ---------------- CONFIG ----------------
# Default to the last training run's best weights; override via --model
DEFAULT_MODEL_PATH = r"D:/Hydrant-sentinel/Hydrant-sentinel/runs/detect/train/weights/best.pt"

# Class mapping from training yaml
CLASS_ID_HYDRANT = 0
CLASS_ID_CAR = 1

# Blockage logic thresholds (tunable)
CONF_THRESHOLD = 0.25
HYDRANT_BUFFER_SCALE_X = 1.6  # expand hydrant bbox horizontally
HYDRANT_BUFFER_SCALE_Y = 1.8  # expand hydrant bbox vertically
CENTER_DISTANCE_FACTOR = 0.8  # consider blocking if centers are close relative to hydrant size


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def expand_box(x1: float, y1: float, x2: float, y2: float, scale_x: float, scale_y: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale_x
    h = (y2 - y1) * scale_y
    nx1 = int(clamp(cx - w / 2.0, 0, img_w - 1))
    ny1 = int(clamp(cy - h / 2.0, 0, img_h - 1))
    nx2 = int(clamp(cx + w / 2.0, 0, img_w - 1))
    ny2 = int(clamp(cy + h / 2.0, 0, img_h - 1))
    return nx1, ny1, nx2, ny2


def boxes_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    return inter_w > 0 and inter_h > 0


def center_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def is_blocking(hydrant_box: Tuple[int, int, int, int], car_box: Tuple[int, int, int, int], img_shape: Tuple[int, int, int]) -> bool:
    img_h, img_w = img_shape[:2]
    # 1) If car intersects an expanded safety zone around hydrant → blocking
    buf_box = expand_box(
        hydrant_box[0], hydrant_box[1], hydrant_box[2], hydrant_box[3],
        HYDRANT_BUFFER_SCALE_X, HYDRANT_BUFFER_SCALE_Y, img_w, img_h,
    )
    if boxes_intersect(buf_box, car_box):
        return True

    # 2) If car center is very close to hydrant center relative to hydrant size → blocking
    hx1, hy1, hx2, hy2 = hydrant_box
    hydrant_size = max(hx2 - hx1, hy2 - hy1)
    if hydrant_size <= 0:
        return False
    dist = center_distance(hydrant_box, car_box)
    return dist < CENTER_DISTANCE_FACTOR * hydrant_size


def draw_box(img, box: Tuple[int, int, int, int], color: Tuple[int, int, int], label: Optional[str] = None):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.rectangle(img, (x1, max(0, y1 - 20)), (x1 + max(80, len(label) * 10), y1), color, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def run_detection(source_path: str, model_path: str = DEFAULT_MODEL_PATH, conf: float = CONF_THRESHOLD, save_dir: Optional[str] = None):
    model = YOLO(model_path)

    paths: List[Path]
    p = Path(source_path)
    if p.is_dir():
        paths = [f for f in p.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    else:
        paths = [p]

    if save_dir is None:
        save_dir = str(Path("runs") / "blockage")
    os.makedirs(save_dir, exist_ok=True)

    for img_path in paths:
        results = model.predict(str(img_path), conf=conf, verbose=False)
        res = results[0]
        img = res.orig_img.copy()
        img_h, img_w = res.orig_shape

        hydrants: List[Tuple[int, int, int, int, float]] = []  # x1,y1,x2,y2,conf
        cars: List[Tuple[int, int, int, int, float]] = []

        for box, cls_id, score in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
            x1, y1, x2, y2 = map(int, box.tolist())
            cid = int(cls_id.item())
            cf = float(score.item()) if score is not None else 1.0
            if cid == CLASS_ID_HYDRANT:
                hydrants.append((x1, y1, x2, y2, cf))
            elif cid == CLASS_ID_CAR:
                cars.append((x1, y1, x2, y2, cf))

        blocking_pairs: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]] = []
        for hx1, hy1, hx2, hy2, _ in hydrants:
            for cx1, cy1, cx2, cy2, _ in cars:
                if is_blocking((hx1, hy1, hx2, hy2), (cx1, cy1, cx2, cy2), img.shape):
                    blocking_pairs.append(((hx1, hy1, hx2, hy2), (cx1, cy1, cx2, cy2)))

        # Draw detections
        for x1, y1, x2, y2, c in hydrants:
            draw_box(img, (x1, y1, x2, y2), (0, 200, 0), f"hydrant {c:.2f}")
        for x1, y1, x2, y2, c in cars:
            draw_box(img, (x1, y1, x2, y2), (0, 140, 255), f"car {c:.2f}")

        # Highlight blocking pairs
        for hbox, cbox in blocking_pairs:
            draw_box(img, hbox, (0, 0, 255), "BLOCKED")
            draw_box(img, cbox, (0, 0, 255), "BLOCKING CAR")

            # Also draw hydrant safety buffer for visualization
            buf = expand_box(hbox[0], hbox[1], hbox[2], hbox[3], HYDRANT_BUFFER_SCALE_X, HYDRANT_BUFFER_SCALE_Y, img_w, img_h)
            cv2.rectangle(img, (buf[0], buf[1]), (buf[2], buf[3]), (0, 0, 255), 1)

        out_path = Path(save_dir) / (img_path.stem + "_blockage.jpg")
        cv2.imwrite(str(out_path), img)

        status = "BLOCKAGE DETECTED" if blocking_pairs else "NO BLOCKAGE"
        print(f"{img_path.name}: hydrants={len(hydrants)} cars={len(cars)} -> {status} -> saved: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect hydrant blockage by parked cars using a trained YOLO model.")
    parser.add_argument("source", type=str, help="Path to image file or directory of images")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model weights (best.pt)")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save annotated outputs")
    parser.add_argument("--bufx", type=float, default=HYDRANT_BUFFER_SCALE_X, help="Hydrant buffer scale X")
    parser.add_argument("--bufy", type=float, default=HYDRANT_BUFFER_SCALE_Y, help="Hydrant buffer scale Y")
    parser.add_argument("--center_factor", type=float, default=CENTER_DISTANCE_FACTOR, help="Center distance factor relative to hydrant size")

    args = parser.parse_args()

    # Allow runtime tuning of thresholds
    global HYDRANT_BUFFER_SCALE_X, HYDRANT_BUFFER_SCALE_Y, CENTER_DISTANCE_FACTOR
    HYDRANT_BUFFER_SCALE_X = args.bufx
    HYDRANT_BUFFER_SCALE_Y = args.bufy
    CENTER_DISTANCE_FACTOR = args.center_factor

    run_detection(args.source, model_path=args.model, conf=args.conf, save_dir=args.save_dir)


