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

# Classes the proposal focuses on
TARGET_CLASSES = {0, 3, 5}  # Car, Pedestrian, Cyclist


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert normalized YOLO bbox to absolute pixel [x1, y1, x2, y2]."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_labels(label_path: Path, img_w: int, img_h: int, target_only: bool = True):
    """
    Load ground-truth boxes from a YOLO-format label file.

    Returns list of dicts: {class_id, class_name, bbox: [x1,y1,x2,y2]}
    If target_only=True, filters to TARGET_CLASSES only.
    """
    annotations = []
    if not label_path.exists():
        return annotations

    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        if target_only and cls_id not in TARGET_CLASSES:
            continue
        cx, cy, w, h = map(float, parts[1:])
        bbox = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
        annotations.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES[cls_id],
            "bbox": bbox,
        })
    return annotations


class KITTIDataset:
    """
    Iterates over a KITTI split (train/val/test).

    Usage:
        dataset = KITTIDataset("datasets/kitti_dataset", split="val")
        for item in dataset:
            image      = item["image"]       # PIL.Image
            image_path = item["image_path"]  # Path
            gt         = item["gt"]          # list of annotation dicts
    """

    def __init__(self, root: str, split: str = "val", target_only: bool = True):
        self.root = Path(root)
        self.split = split
        self.target_only = target_only

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
        gt = load_labels(label_path, img_w, img_h, self.target_only)

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
    dataset = KITTIDataset("datasets/kitti_dataset", split="train")
    print(f"Total images: {len(dataset)}")

    sample = dataset[0]
    print(f"Image: {sample['image_path'].name}  size: {sample['img_w']}x{sample['img_h']}")
    print(f"Ground truth ({len(sample['gt'])} objects):")
    for ann in sample["gt"]:
        bbox = [f"{v:.1f}" for v in ann["bbox"]]
        print(f"  {ann['class_name']:15s} bbox={bbox}")
