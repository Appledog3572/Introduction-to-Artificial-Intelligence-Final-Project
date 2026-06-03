"""
Evaluate LLaVA baseline on KITTI dataset.

Metrics:
  - mAP@0.5        (per-class and overall)
  - Mean FPS       (1000 / mean_latency_ms)
  - Mean latency   (ms per image)
  - Model size     (MB)

mAP is computed with a pure-Python IoU matching implementation
(no external dependencies like pycocotools).

Usage:
    python evaluate.py --split val --mode mock --max-images 50
    python evaluate.py --split val --mode llava
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from dataset import KITTIDataset, TARGET_CLASSES, CLASS_NAMES
from infer import InferenceRunner


# ---------------------------------------------------------------------------
# IoU and AP helpers
# ---------------------------------------------------------------------------

def compute_iou(boxA: list[float], boxB: list[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def compute_ap(precisions: list[float], recalls: list[float]) -> float:
    """
    Compute Average Precision using the 11-point interpolation method
    (standard for PASCAL VOC / KITTI).
    """
    ap = 0.0
    for thr in [r / 10 for r in range(11)]:
        prec_at_thr = [p for p, r in zip(precisions, recalls) if r >= thr]
        ap += max(prec_at_thr, default=0.0)
    return ap / 11


def compute_ap_for_class(
    all_preds: list[dict],   # [{score, bbox, image_id}]
    all_gts: list[dict],     # [{bbox, image_id, matched}]
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute AP for a single class.

    all_preds: sorted by score descending (done inside this function)
    all_gts:   ground-truth boxes for this class
    """
    if not all_gts:
        return 0.0

    all_preds = sorted(all_preds, key=lambda x: x["score"], reverse=True)

    # Group GTs by image
    gt_by_image: dict[str, list[dict]] = defaultdict(list)
    for gt in all_gts:
        gt_by_image[gt["image_id"]].append({**gt, "matched": False})

    tp = []
    fp = []
    for pred in all_preds:
        img_id = pred["image_id"]
        gts = gt_by_image.get(img_id, [])

        best_iou = iou_threshold - 1e-9
        best_idx = -1
        for idx, gt in enumerate(gts):
            if gt["matched"]:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0:
            gts[best_idx]["matched"] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    cum_tp = []
    cum_fp = []
    running_tp = running_fp = 0
    for t, f in zip(tp, fp):
        running_tp += t
        running_fp += f
        cum_tp.append(running_tp)
        cum_fp.append(running_fp)

    n_gt = len(all_gts)
    precisions = [t / (t + f) for t, f in zip(cum_tp, cum_fp)]
    recalls    = [t / n_gt for t in cum_tp]

    return compute_ap(precisions, recalls)


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
) -> dict:

    dataset = KITTIDataset("datasets/kitti_dataset", split=split)
    runner  = InferenceRunner(mode=mode, model_id=model_id, api_key=api_key)

    n = min(len(dataset), max_images) if max_images else len(dataset)
    print(f"Evaluating {n} images  [mode={mode}  split={split}]")

    # Accumulators keyed by class_id
    preds_by_class: dict[int, list] = defaultdict(list)
    gts_by_class:   dict[int, list] = defaultdict(list)

    latencies: list[float] = []
    per_image: list[dict] = []

    for i in range(n):
        item   = dataset[i]
        image  = item["image"]
        img_id = item["image_path"].stem

        t0     = time.perf_counter()
        result = runner.run(image)
        wall_ms = (time.perf_counter() - t0) * 1000

        # Use model-reported latency when available, else wall-clock
        latency = result.latency_ms if result.latency_ms > 0 else wall_ms
        latencies.append(latency)

        preds = [
            {"class": d.class_name, "bbox": d.bbox, "score": d.score}
            for d in result.detections
        ]
        gts = [
            {"class": g["class_name"], "bbox": g["bbox"]}
            for g in item["gt"]
        ]

        per_image.append({
            "image_id":   img_id,
            "latency_ms": round(latency, 1),
            "raw_text":   result.raw_text,
            "predictions": preds,
            "ground_truth": gts,
        })

        for det in result.detections:
            preds_by_class[det.class_id].append({
                "score":    det.score,
                "bbox":     det.bbox,
                "image_id": img_id,
            })

        for gt in item["gt"]:
            gts_by_class[gt["class_id"]].append({
                "bbox":     gt["bbox"],
                "image_id": img_id,
            })

        if debug_n > 0 and i < debug_n:
            print(f"\n[DEBUG {item['image_path'].name}]")
            print(f"  raw_text : {result.raw_text!r}")
            print(f"  parsed   : {result.detections}")

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n} images processed …")

    # Per-class AP
    ap_per_class: dict[str, float] = {}
    for cls_id in TARGET_CLASSES:
        ap = compute_ap_for_class(
            preds_by_class[cls_id],
            gts_by_class[cls_id],
            iou_threshold,
        )
        ap_per_class[CLASS_NAMES[cls_id]] = round(ap, 4)

    mean_ap   = round(sum(ap_per_class.values()) / len(ap_per_class), 4)
    mean_lat  = round(sum(latencies) / len(latencies), 2)
    mean_fps  = round(1000 / mean_lat, 2)
    model_mb  = round(runner.model_size_mb, 1)

    results = {
        "mode":            mode,
        "split":           split,
        "n_images":        n,
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
        Path(output_json).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {output_json}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",       default="val",  choices=["train", "val", "test"])
    parser.add_argument("--mode",        default="mock", choices=["mock", "llava", "gemini", "gpt4o"])
    parser.add_argument("--model-id",    default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--max-images",  type=int, default=None)
    parser.add_argument("--iou",         type=float, default=0.5)
    parser.add_argument("--output-json", default=None,
                        help="Path to save results as JSON")
    parser.add_argument("--debug-n", type=int, default=0,
                        help="Print raw model output for first N images")
    parser.add_argument("--api-key", default=None,
                        help="API key (gemini mode); falls back to GEMINI_API_KEY env var")
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
    )
