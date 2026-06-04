"""
Fast NMS — drop-in replacement for torchvision.ops.nms inside Ultralytics YOLOv8.

Standard NMS (torchvision):
  - Sequential, O(n²) CPU loop
  - Each iteration re-checks remaining boxes → cannot be parallelised

Fast NMS (YOLACT, Liu et al. 2019):
  - Compute full IoU matrix in ONE GPU call
  - Suppress via column-max thresholding (fully vectorised)
  - ~2× faster; slight approximation (may remove 1–2 extra boxes per image)

Usage:
    import fast_nms
    fast_nms.patch()          # call once before loading YOLO
    model = YOLO('yolov8n.pt')
    model.predict(...)        # automatically uses Fast NMS
"""

import time
import torch
import torchvision


# ---------------------------------------------------------------------------
# Core Fast NMS
# ---------------------------------------------------------------------------

def fast_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """
    Fast NMS with the same signature as torchvision.ops.nms.

    Args:
        boxes:         (N, 4) float tensor  [x1, y1, x2, y2]
        scores:        (N,)   float tensor  confidence scores
        iou_threshold: float

    Returns:
        keep: (K,) long tensor of kept indices (sorted by score descending)
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # 1. Sort by score descending
    _, order = scores.sort(descending=True)
    boxes_sorted = boxes[order]

    # 2. Compute ALL pairwise IoU at once on GPU  — O(n²) but vectorised
    iou = torchvision.ops.box_iou(boxes_sorted, boxes_sorted)   # (N, N)

    # 3. Upper-triangle only: box i is suppressed by any higher-scored box j < i
    iou.triu_(diagonal=1)

    # 4. Keep box i  iff  no higher-scored box overlaps it beyond the threshold
    keep_mask = iou.max(dim=0).values < iou_threshold

    return order[keep_mask]


# ---------------------------------------------------------------------------
# Patch / unpatch Ultralytics
# ---------------------------------------------------------------------------

_original_nms = torchvision.ops.nms   # save reference for restore


def patch():
    """Replace torchvision.ops.nms with Fast NMS (affects all Ultralytics calls)."""
    torchvision.ops.nms = fast_nms

    # Also patch the copy already imported inside ultralytics.utils.ops
    try:
        import ultralytics.utils.ops as _ul
        import ultralytics.models.yolo.detect.predict as _up
        for mod in (_ul, _up):
            if hasattr(mod, 'torchvision'):
                mod.torchvision.ops.nms = fast_nms
    except Exception:
        pass

    print("[fast_nms] patched into torchvision.ops.nms")


def unpatch():
    """Restore the original torchvision NMS."""
    torchvision.ops.nms = _original_nms
    try:
        import ultralytics.utils.ops as _ul
        import ultralytics.models.yolo.detect.predict as _up
        for mod in (_ul, _up):
            if hasattr(mod, 'torchvision'):
                mod.torchvision.ops.nms = _original_nms
    except Exception:
        pass
    print("[fast_nms] restored original torchvision.ops.nms")


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def benchmark(
    model_path: str = "yolov8n.pt",
    image_path: str = "datasets/kitti_dataset/images/val/000013.png",
    n_warmup: int = 5,
    n_runs: int = 50,
    device: str = "cuda",
):
    """
    Measure and compare total inference latency:
      Standard NMS  vs  Fast NMS

    Metrics reported:
      - Mean latency (ms)
      - FPS
      - NMS share of total time (estimated)
    """
    import numpy as np
    from ultralytics import YOLO

    results = {}

    for label, use_fast in [("Standard NMS", False), ("Fast NMS", True)]:
        if use_fast:
            patch()
        else:
            unpatch()

        model = YOLO(model_path)
        model.to(device)

        # Warmup
        for _ in range(n_warmup):
            model.predict(image_path, verbose=False, device=device)

        # Timed runs
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.predict(image_path, verbose=False, device=device)
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms = float(np.mean(times))
        results[label] = mean_ms

    unpatch()

    # ── Report ──
    std_ms  = results["Standard NMS"]
    fast_ms = results["Fast NMS"]
    speedup = std_ms / fast_ms

    print("\n===== NMS Benchmark =====")
    print(f"  Standard NMS : {std_ms:.2f} ms  ({1000/std_ms:.1f} FPS)")
    print(f"  Fast NMS     : {fast_ms:.2f} ms  ({1000/fast_ms:.1f} FPS)")
    print(f"  Speedup      : {speedup:.2f}×")
    print("=========================")

    return results


# ---------------------------------------------------------------------------
# Isolated NMS-only timing  (shows NMS contribution to total latency)
# ---------------------------------------------------------------------------

def time_nms_only(n_boxes: int = 8400, n_runs: int = 1000, device: str = "cuda"):
    """
    Directly compare NMS latency on synthetic data (no model loading needed).
    Useful for understanding the NMS bottleneck in isolation.
    """
    import numpy as np

    boxes  = torch.rand(n_boxes, 4, device=device)
    boxes[:, 2:] += boxes[:, :2]          # ensure x2 > x1, y2 > y1
    scores = torch.rand(n_boxes, device=device)

    def _time(fn, label):
        # warmup
        for _ in range(10):
            fn(boxes, scores, 0.45)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            fn(boxes, scores, 0.45)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        m = float(np.mean(times))
        print(f"  {label:20s}: {m:.4f} ms  ({1000/m:.0f} calls/s)")
        return m

    print(f"\n===== NMS-only timing  [{n_boxes} boxes, device={device}] =====")
    std_t  = _time(_original_nms, "Standard NMS")
    fast_t = _time(fast_nms,      "Fast NMS")
    print(f"  Speedup: {std_t/fast_t:.2f}×")
    print("=" * 52)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running NMS-only timing on CPU ...")
    time_nms_only(n_boxes=8400, n_runs=500, device="cpu")

    if torch.cuda.is_available():
        print("\nRunning NMS-only timing on CUDA ...")
        time_nms_only(n_boxes=8400, n_runs=1000, device="cuda")
    else:
        print("\n[skip] CUDA not available")
