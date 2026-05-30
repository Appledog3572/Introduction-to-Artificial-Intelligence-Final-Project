"""
Parse LLaVA text output into structured bounding box detections.

Expected LLaVA prompt instructs output in the format:
    <class> <x1> <y1> <x2> <y2>
one object per line, coordinates in pixels.

Example valid output:
    Car 120 85 340 210
    Pedestrian 500 100 530 195
    Cyclist 700 90 760 200

The parser is tolerant of:
- Extra punctuation / colons (e.g. "Car: 120 85 340 210")
- Mixed case class names
- Lines that don't match (silently skipped)
- Confidence scores if the model appends them (e.g. "Car 120 85 340 210 0.92")
"""

import re
from dataclasses import dataclass

from dataset import CLASS_NAMES, TARGET_CLASSES


# Map lowercase variants → canonical class_id
_NAME_TO_ID: dict[str, int] = {}
for _cid, _cname in CLASS_NAMES.items():
    _NAME_TO_ID[_cname.lower()] = _cid
# Common aliases LLaVA might use
_ALIASES = {
    "person": 3,        # → Pedestrian
    "people": 3,
    "human": 3,
    "vehicle": 0,       # → Car
    "automobile": 0,
    "truck": 2,
    "van": 1,
    "bike": 5,          # → Cyclist
    "bicycle": 5,
    "motorcyclist": 5,
}
_NAME_TO_ID.update(_ALIASES)

# Regex: class name, then 4 numbers (optionally wrapped in <>), optional confidence
# Handles both "Car 120 85 340 210" and "Car <120> <85> <340> <210>"
_NUM = r"<?\s*(-?\d+(?:\.\d+)?)\s*>?"   # number optionally inside < >
_LINE_RE = re.compile(
    r"([a-zA-Z_]+)[\s:,]*"   # class name
    + r"\s+".join([_NUM] * 4)  # x1 y1 x2 y2
    + r"(?:\s+" + _NUM + r")?",  # optional confidence
)


@dataclass
class Detection:
    class_id: int
    class_name: str
    bbox: list[float]   # [x1, y1, x2, y2] in pixels
    score: float        # confidence; 1.0 if not provided by model


def parse_llava_output(
    text: str,
    img_w: int,
    img_h: int,
    target_only: bool = True,
    min_score: float = 0.0,
) -> list[Detection]:
    """
    Parse raw LLaVA text into a list of Detection objects.

    Args:
        text:        Raw string from LLaVA.
        img_w/h:     Image dimensions — used to clamp coords to valid range.
        target_only: If True, keep only Car / Pedestrian / Cyclist.
        min_score:   Discard detections below this confidence (if provided).
    """
    detections = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = _LINE_RE.search(line)
        if not m:
            continue

        raw_class = m.group(1).lower()
        class_id = _NAME_TO_ID.get(raw_class)
        if class_id is None:
            continue
        if target_only and class_id not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = (float(m.group(i)) for i in range(2, 6))
        score = float(m.group(6)) if m.group(6) is not None else 1.0

        # If coordinates are normalized (all values <= 1.0), convert to pixels
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
            x1, x2 = x1 * img_w, x2 * img_w
            y1, y2 = y1 * img_h, y2 * img_h

        if score < min_score:
            continue

        # Ensure x1 < x2, y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Clamp to image bounds
        x1 = max(0.0, min(x1, img_w))
        x2 = max(0.0, min(x2, img_w))
        y1 = max(0.0, min(y1, img_h))
        y2 = max(0.0, min(y2, img_h))

        # Skip degenerate boxes
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(Detection(
            class_id=class_id,
            class_name=CLASS_NAMES[class_id],
            bbox=[x1, y1, x2, y2],
            score=score,
        ))

    return detections


def build_prompt(img_w: int, img_h: int) -> str:
    """Return the prompt sent to LLaVA for object detection."""
    return (
        "You are an object detection system for autonomous driving. "
        "Detect all objects in this image from these classes: Car, Pedestrian, Cyclist.\n\n"
        f"Image size: {img_w} x {img_h} pixels.\n\n"
        "For each detected object, output exactly one line using this format:\n"
        "CLASS X1 Y1 X2 Y2\n\n"
        "Rules:\n"
        "- CLASS is one of: Car, Pedestrian, Cyclist\n"
        "- X1 Y1 is the top-left corner in pixels (integers)\n"
        "- X2 Y2 is the bottom-right corner in pixels (integers)\n"
        "- Do not use brackets, colons, or any extra text\n\n"
        "Example output:\n"
        "Car 245 120 480 310\n"
        "Pedestrian 530 95 560 200\n\n"
        "Now detect all objects in the image:"
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    MOCK_OUTPUT = """
    Car 120 85 340 210
    Pedestrian: 500 100 530 195
    cyclist 700 90 760 200 0.87
    Van 50 60 200 180
    unknown_object 10 10 50 50
    Car 999 999 1100 1100
    """

    IMG_W, IMG_H = 1242, 375
    results = parse_llava_output(MOCK_OUTPUT, IMG_W, IMG_H)

    print(f"Parsed {len(results)} detections (target classes only):\n")
    for d in results:
        print(f"  [{d.class_id}] {d.class_name:15s}  bbox={[f'{v:.1f}' for v in d.bbox]}  score={d.score:.2f}")

    print()
    print("Sample prompt:")
    print(build_prompt(IMG_W, IMG_H))
