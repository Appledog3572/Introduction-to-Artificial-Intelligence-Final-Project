"""
Visualize LLaVA predictions and ground-truth boxes on KITTI images.

Reads the per-image JSON output from evaluate.py and draws:
  - Green boxes : ground truth
  - Red boxes   : LLaVA predictions

Usage:
    python llava_baseline/visualize.py \
        --json results/llava_val_100.json \
        --image-dir datasets/kitti_dataset/images/val \
        --output-dir results/vis \
        --max-images 20
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Colours
GT_COLOR   = (0, 200, 0)    # green  — ground truth
PRED_COLOR = (220, 30, 30)  # red    — prediction
LINE_WIDTH = 3

CLASS_COLORS = {
    "Car":        (255, 100, 0),
    "Pedestrian": (0, 150, 255),
    "Cyclist":    (200, 0, 200),
}


def draw_box(draw: ImageDraw.Draw, bbox: list, label: str, color: tuple, is_gt: bool):
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=LINE_WIDTH)

    prefix = "GT" if is_gt else "PR"
    text   = f"{prefix}:{label}"

    # Label background
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    bbox_text = draw.textbbox((x1, y1), text, font=font)
    tw = bbox_text[2] - bbox_text[0]
    th = bbox_text[3] - bbox_text[1]

    ty = y1 - th - 4 if y1 - th - 4 >= 0 else y1 + 2
    draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
    draw.text((x1 + 2, ty + 2), text, fill=(255, 255, 255), font=font)


def visualize(
    json_path: str,
    image_dir: str,
    output_dir: str,
    max_images: int | None = None,
    only_with_preds: bool = False,
):
    data       = json.loads(Path(json_path).read_text())
    per_image  = data.get("per_image", [])
    image_dir  = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for item in per_image:
        if max_images and saved >= max_images:
            break
        if only_with_preds and not item["predictions"]:
            continue

        img_path = image_dir / (item["image_id"] + ".png")
        if not img_path.exists():
            print(f"  [skip] image not found: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        draw  = ImageDraw.Draw(image)

        for gt in item["ground_truth"]:
            color = CLASS_COLORS.get(gt["class"], GT_COLOR)
            draw_box(draw, gt["bbox"], gt["class"], GT_COLOR, is_gt=True)

        for pred in item["predictions"]:
            draw_box(draw, pred["bbox"], pred["class"], PRED_COLOR, is_gt=False)

        # Caption at bottom
        caption = f"PR:{len(item['predictions'])}  GT:{len(item['ground_truth'])}  {item['latency_ms']}ms"
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        draw.text((6, image.height - 20), caption, fill=(255, 255, 0), font=font)

        out_path = output_dir / (item["image_id"] + ".png")
        image.save(out_path)
        saved += 1

    print(f"Saved {saved} images to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",        required=True,  help="Path to evaluate.py output JSON")
    parser.add_argument("--image-dir",   required=True,  help="Directory containing original images")
    parser.add_argument("--output-dir",  default="results/vis")
    parser.add_argument("--max-images",  type=int, default=None)
    parser.add_argument("--only-with-preds", action="store_true",
                        help="Only save images that have at least one prediction")
    args = parser.parse_args()

    visualize(
        json_path=args.json,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
        only_with_preds=args.only_with_preds,
    )
