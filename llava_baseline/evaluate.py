"""
Evaluate LLaVA baseline on KITTI dataset.

Metrics:
  - mAP@0.5        (per-class and overall)
  - Mean FPS / latency
  - Model size (MB)
  - Confusion matrix (saved as PNG)

Usage:
    python evaluate.py --split val --mode mock --max-images 50
    python evaluate.py --split val --mode gemini --api-key KEY \\
        --full-classes --vis-dir results/vis --output-json results/out.json
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from dataset import KITTIDataset, TARGET_CLASSES, FULL_CLASSES, CLASS_NAMES
from infer import InferenceRunner


# ---------------------------------------------------------------------------
# IoU and AP helpers
# ---------------------------------------------------------------------------

def compute_iou(boxA: list[float], boxB: list[float]) -> float:
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def compute_ap(precisions, recalls):
    ap = 0.0
    for thr in [r / 10 for r in range(11)]:
        prec_at_thr = [p for p, r in zip(precisions, recalls) if r >= thr]
        ap += max(prec_at_thr, default=0.0)
    return ap / 11


def compute_ap_for_class(all_preds, all_gts, iou_threshold=0.5):
    if not all_gts:
        return 0.0
    all_preds = sorted(all_preds, key=lambda x: x["score"], reverse=True)
    gt_by_image = defaultdict(list)
    for gt in all_gts:
        gt_by_image[gt["image_id"]].append({**gt, "matched": False})

    tp, fp = [], []
    for pred in all_preds:
        gts = gt_by_image.get(pred["image_id"], [])
        best_iou, best_idx = iou_threshold - 1e-9, -1
        for idx, gt in enumerate(gts):
            if gt["matched"]:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_idx >= 0:
            gts[best_idx]["matched"] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)

    cum_tp, cum_fp, rt, rf = [], [], 0, 0
    for t, f in zip(tp, fp):
        rt += t; rf += f
        cum_tp.append(rt); cum_fp.append(rf)

    n_gt = len(all_gts)
    precs = [t / (t + f) for t, f in zip(cum_tp, cum_fp)]
    recs  = [t / n_gt    for t in cum_tp]
    return compute_ap(precs, recs)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def compute_confusion_matrix(per_image_data, class_ids, iou_threshold=0.5):
    """
    Build a (n_classes+1) x (n_classes+1) confusion matrix.
    Last row/col = "background" (FN / FP).
    Axes: rows = Predicted, cols = True  (matching the example image).
    """
    cls_list   = sorted(class_ids)
    cls_to_idx = {c: i for i, c in enumerate(cls_list)}
    n          = len(cls_list)
    bg         = n
    matrix     = np.zeros((n + 1, n + 1), dtype=int)

    # Build per-image lookup
    gts_by_img   = defaultdict(list)
    preds_by_img = defaultdict(list)
    for item in per_image_data:
        img_id = item["image_id"]
        for g in item["ground_truth"]:
            gts_by_img[img_id].append(g)
        for p in item["predictions"]:
            preds_by_img[img_id].append(p)

    for img_id, gts in gts_by_img.items():
        preds = sorted(preds_by_img.get(img_id, []),
                       key=lambda x: x.get("score", 1.0), reverse=True)
        gt_matched = [False] * len(gts)

        for pred in preds:
            pred_cls_id  = pred.get("class_id", _name_to_id(pred.get("class", "")))
            pred_cls_idx = cls_to_idx.get(pred_cls_id, bg)

            best_iou, best_j = iou_threshold - 1e-9, -1
            for j, gt in enumerate(gts):
                if gt_matched[j]:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_j >= 0:
                gt_matched[best_j] = True
                gt_cls_id  = gts[best_j].get("class_id",
                                 _name_to_id(gts[best_j].get("class", "")))
                gt_cls_idx = cls_to_idx.get(gt_cls_id, bg)
                matrix[pred_cls_idx][gt_cls_idx] += 1   # predicted row, true col
            else:
                matrix[pred_cls_idx][bg] += 1            # FP

        for j, gt in enumerate(gts):
            if not gt_matched[j]:
                gt_cls_id  = gt.get("class_id", _name_to_id(gt.get("class", "")))
                gt_cls_idx = cls_to_idx.get(gt_cls_id, bg)
                matrix[bg][gt_cls_idx] += 1              # FN

    return matrix, cls_list


def _name_to_id(name: str) -> int:
    """Reverse lookup: class name → class_id."""
    for cid, cname in CLASS_NAMES.items():
        if cname.lower() == name.lower():
            return cid
    return -1


def save_confusion_matrix(matrix, cls_list, output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels  = [CLASS_NAMES.get(c, str(c)) for c in cls_list] + ["background"]
    n       = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n), max(7, n - 1)))

    vmax = matrix.max() if matrix.max() > 0 else 1
    im   = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("True");   ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix")

    thresh = vmax * 0.5
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > thresh else "black", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


# ---------------------------------------------------------------------------
# Per-image visualization helper
# ---------------------------------------------------------------------------

def _save_vis(image, gt_list, pred_list, out_path: Path):
    from PIL import ImageDraw, ImageFont
    img  = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    GT_COLOR   = (0, 200, 0)
    PRED_COLOR = (220, 30, 30)

    for objs, color, prefix in [(gt_list, GT_COLOR, "GT"), (pred_list, PRED_COLOR, "PR")]:
        for obj in objs:
            x1, y1, x2, y2 = obj["bbox"]
            cid   = obj.get("class_id", _name_to_id(obj.get("class", "")))
            label = f"{prefix}:{cid}"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            bb = draw.textbbox((x1, y1), label, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            ty = y1 - th - 3 if y1 - th - 3 >= 0 else y1 + 2
            draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
            draw.text((x1 + 2, ty + 2), label, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    split: str = "val",
    mode: str = "mock",
    model_id: str | None = None,
    max_images: int | None = None,
    iou_threshold: float = 0.5,
    output_json: str | None = None,
    debug_n: int = 0,
    api_key: str | None = None,
    full_classes: bool = False,
    vis_dir: str | None = None,
) -> dict:

    active_classes = FULL_CLASSES if full_classes else TARGET_CLASSES
    dataset = KITTIDataset("datasets/kitti_dataset", split=split,
                           full_classes=full_classes)
    runner  = InferenceRunner(mode=mode, model_id=model_id, api_key=api_key)

    n = min(len(dataset), max_images) if max_images else len(dataset)
    print(f"Evaluating {n} images  "
          f"[mode={mode}  split={split}  "
          f"classes={'full(8)' if full_classes else 'target(3)'}]")

    preds_by_class: dict[int, list] = defaultdict(list)
    gts_by_class:   dict[int, list] = defaultdict(list)
    latencies: list[float] = []
    per_image: list[dict]  = []

    vis_path = Path(vis_dir) if vis_dir else None
    if vis_path:
        vis_path.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        item   = dataset[i]
        image  = item["image"]
        img_id = item["image_path"].stem

        t0      = time.perf_counter()
        result  = runner.run(image)
        wall_ms = (time.perf_counter() - t0) * 1000

        latency = result.latency_ms if result.latency_ms > 0 else wall_ms
        latencies.append(latency)

        preds = [{"class": d.class_name, "class_id": d.class_id,
                  "bbox": d.bbox, "score": d.score}
                 for d in result.detections]
        gts   = [{"class": g["class_name"], "class_id": g["class_id"],
                  "bbox": g["bbox"]}
                 for g in item["gt"]]

        per_image.append({
            "image_id":     img_id,
            "latency_ms":   round(latency, 1),
            "raw_text":     result.raw_text,
            "predictions":  preds,
            "ground_truth": gts,
        })

        for det in result.detections:
            preds_by_class[det.class_id].append(
                {"score": det.score, "bbox": det.bbox, "image_id": img_id})
        for gt in item["gt"]:
            gts_by_class[gt["class_id"]].append(
                {"bbox": gt["bbox"], "image_id": img_id})

        # Save visualization
        if vis_path:
            _save_vis(image, gts, preds, vis_path / f"{img_id}.png")

        if debug_n > 0 and i < debug_n:
            print(f"\n[DEBUG {item['image_path'].name}]")
            print(f"  raw_text : {result.raw_text!r}")
            print(f"  parsed   : {result.detections}")

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n} images processed …")

    # Per-class AP
    ap_per_class: dict[str, float] = {}
    for cls_id in active_classes:
        ap = compute_ap_for_class(
            preds_by_class[cls_id], gts_by_class[cls_id], iou_threshold)
        ap_per_class[CLASS_NAMES[cls_id]] = round(ap, 4)

    mean_ap  = round(sum(ap_per_class.values()) / max(len(ap_per_class), 1), 4)
    mean_lat = round(sum(latencies) / len(latencies), 2)
    mean_fps = round(1000 / mean_lat, 2)
    model_mb = round(runner.model_size_mb, 1)

    results = {
        "mode":            mode,
        "split":           split,
        "n_images":        n,
        "full_classes":    full_classes,
        "iou_threshold":   iou_threshold,
        "mAP":             mean_ap,
        "AP_per_class":    ap_per_class,
        "mean_latency_ms": mean_lat,
        "mean_FPS":        mean_fps,
        "model_size_MB":   model_mb,
        "per_image":       per_image,
    }

    print("\n=== Results ===")
    print(f"  mAP@{iou_threshold}:        {mean_ap:.4f}")
    for cls, ap in ap_per_class.items():
        print(f"    {cls:15s}  AP={ap:.4f}")
    print(f"  Mean latency:  {mean_lat:.1f} ms")
    print(f"  Mean FPS:      {mean_fps:.2f}")
    print(f"  Model size:    {model_mb:.1f} MB")

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {output_json}")

        # Confusion matrix (saved alongside JSON)
        cm, cls_list = compute_confusion_matrix(per_image, active_classes, iou_threshold)
        cm_path = Path(output_json).with_suffix("") .parent / \
                  (Path(output_json).stem + "_confusion.png")
        save_confusion_matrix(cm, cls_list, str(cm_path))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",        default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--mode",         default="mock",
                        choices=["mock", "llava", "gemini", "gpt4o"])
    parser.add_argument("--model-id",     default=None)
    parser.add_argument("--max-images",   type=int,   default=None)
    parser.add_argument("--iou",          type=float, default=0.5)
    parser.add_argument("--output-json",  default=None)
    parser.add_argument("--debug-n",      type=int,   default=0)
    parser.add_argument("--api-key",      default=None)
    parser.add_argument("--full-classes", action="store_true",
                        help="Evaluate all 8 KITTI classes instead of 3")
    parser.add_argument("--vis-dir",      default=None,
                        help="Save annotated images here (GT=green, Pred=red)")
    args = parser.parse_args()

    evaluate(
        split=args.split,
        mode=args.mode,
        model_id=args.model_id,
        max_images=args.max_images,
        iou_threshold=args.iou,
        output_json=args.output_json,
        debug_n=args.debug_n,
        api_key=args.api_key,
        full_classes=args.full_classes,
        vis_dir=args.vis_dir,
    )
