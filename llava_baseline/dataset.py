"""
KITTI dataset loader for LLaVA baseline evaluation.
Labels are in YOLO format: class cx cy w h (normalized).
"""

from pathlib import Path
from PIL import Image


CLASS_NAMES = {
    0: "Car",
    1: "Van",
    2: "Truck",
    3: "Pedestrian",
    4: "Person_sitting",
    5: "Cyclist",
    6: "Tram",
    7: "Misc",
    8: "DontCare",
}

# Proposal focus: Car, Pedestrian, Cyclist
TARGET_CLASSES = {0, 3, 5}

# Full KITTI classes (excluding DontCare=8)
FULL_CLASSES = {0, 1, 2, 3, 4, 5, 6, 7}


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert normalized YOLO bbox to absolute pixel [x1, y1, x2, y2]."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_labels(label_path: Path, img_w: int, img_h: int,
                allowed_classes: set | None = None):
    """
    Load ground-truth boxes from a YOLO-format label file.

    allowed_classes: set of class IDs to keep (None → TARGET_CLASSES).
    """
    if allowed_classes is None:
        allowed_classes = TARGET_CLASSES
    annotations = []
    if not label_path.exists():
        return annotations

    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        if cls_id not in allowed_classes:
            continue
        cx, cy, w, h = map(float, parts[1:])
        bbox = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
        annotations.append({
            "class_id":   cls_id,
            "class_name": CLASS_NAMES[cls_id],
            "bbox":       bbox,
        })
    return annotations


class KITTIDataset:
    """
    Iterates over a KITTI split (train/val/test).

    Args:
        full_classes: If True, load all 8 KITTI classes; otherwise only
                      Car / Pedestrian / Cyclist (TARGET_CLASSES).
    """

    def __init__(self, root: str, split: str = "val", full_classes: bool = False):
        self.root = Path(root)
        self.split = split
        self.allowed_classes = FULL_CLASSES if full_classes else TARGET_CLASSES

        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        label_path = self.label_dir / (image_path.stem + ".txt")
        gt = load_labels(label_path, img_w, img_h, self.allowed_classes)

        return {
            "image": image,
            "image_path": image_path,
            "gt": gt,
            "img_w": img_w,
            "img_h": img_h,
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


if __name__ == "__main__":
    for label in ("3-class", "full"):
        full = label == "full"
        dataset = KITTIDataset("datasets/kitti_dataset", split="train", full_classes=full)
        sample = dataset[0]
        print(f"[{label}] {sample['image_path'].name}  gt={len(sample['gt'])} objects")
        for ann in sample["gt"]:
            print(f"  [{ann['class_id']}] {ann['class_name']}")
