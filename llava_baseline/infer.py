"""
LLaVA inference wrapper for KITTI object detection.

Two modes:
  - mock:  Returns random detections; lets you test the full pipeline
           without loading the model.
  - llava: Loads LLaVA-1.5-7B (or another HuggingFace model) and runs
           real inference. Requires a GPU with ~8 GB VRAM (4-bit quant).

Usage:
    runner = InferenceRunner(mode="mock")
    result = runner.run(image, img_w, img_h)
    print(result.detections, result.latency_ms)
"""

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from parse_output import Detection, build_prompt, parse_llava_output
from dataset import TARGET_CLASSES, CLASS_NAMES


@dataclass
class InferenceResult:
    detections: list[Detection]
    raw_text: str           # raw LLaVA output (empty in mock mode)
    latency_ms: float       # wall-clock time for model inference only


# ---------------------------------------------------------------------------
# Mock runner — no model required
# ---------------------------------------------------------------------------

class MockRunner:
    """
    Simulates LLaVA output with random boxes for pipeline testing.
    Latency is artificially set to represent what a real LLM would cost.
    """

    def __init__(self, fake_latency_ms: float = 5000.0, seed: int = 42):
        self._rng = random.Random(seed)
        self.fake_latency_ms = fake_latency_ms

    def run(self, image: Image.Image) -> InferenceResult:
        img_w, img_h = image.size
        n = self._rng.randint(0, 4)
        detections = []
        for _ in range(n):
            cls_id = self._rng.choice(list(TARGET_CLASSES))
            x1 = self._rng.randint(0, img_w - 50)
            y1 = self._rng.randint(0, img_h - 50)
            x2 = self._rng.randint(x1 + 20, min(x1 + 200, img_w))
            y2 = self._rng.randint(y1 + 20, min(y1 + 100, img_h))
            detections.append(Detection(
                class_id=cls_id,
                class_name=CLASS_NAMES[cls_id],
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                score=round(self._rng.uniform(0.5, 1.0), 2),
            ))
        return InferenceResult(
            detections=detections,
            raw_text="",
            latency_ms=self.fake_latency_ms,
        )


# ---------------------------------------------------------------------------
# Real LLaVA runner
# ---------------------------------------------------------------------------

class LLaVARunner:
    """
    Loads LLaVA-1.5-7B with 4-bit quantization and runs inference.

    Requires:
        pip install transformers accelerate bitsandbytes pillow
    """

    DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"

    def __init__(self, model_id: str = DEFAULT_MODEL):
        # Imports deferred so mock mode works without these packages
        from transformers import (
            LlavaForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        import torch

        print(f"Loading {model_id} with 4-bit quantization …")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant_cfg,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()
        print("Model ready.")

    def run(self, image: Image.Image) -> InferenceResult:
        import torch

        img_w, img_h = image.size
        prompt_text = build_prompt(img_w, img_h)

        # LLaVA-1.5 conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.model.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        # Decode only the newly generated tokens
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw_text = self.processor.decode(generated, skip_special_tokens=True)

        detections = parse_llava_output(raw_text, img_w, img_h)
        return InferenceResult(
            detections=detections,
            raw_text=raw_text,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

class InferenceRunner:
    """
    Factory that returns the right runner based on mode.

    Args:
        mode:     "mock" or "llava"
        model_id: HuggingFace model ID (llava mode only)
    """

    def __init__(self, mode: str = "mock", model_id: str = LLaVARunner.DEFAULT_MODEL):
        if mode == "mock":
            self._runner = MockRunner()
        elif mode == "llava":
            self._runner = LLaVARunner(model_id)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'mock' or 'llava'.")
        self.mode = mode

    def run(self, image: Image.Image) -> InferenceResult:
        return self._runner.run(image)

    @property
    def model_size_mb(self) -> float:
        """Return model size in MB (0.0 for mock mode)."""
        if self.mode == "mock":
            return 0.0
        import torch
        model = self._runner.model
        total = sum(p.numel() * p.element_size() for p in model.parameters())
        return total / (1024 ** 2)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dataset import KITTIDataset

    dataset = KITTIDataset("datasets/kitti_dataset", split="train")
    runner = InferenceRunner(mode="mock")

    print("Running mock inference on 3 images:\n")
    for i in range(3):
        item = dataset[i]
        result = runner.run(item["image"])
        print(f"[{item['image_path'].name}]")
        print(f"  GT:        {len(item['gt'])} objects")
        print(f"  Predicted: {len(result.detections)} objects")
        print(f"  Latency:   {result.latency_ms:.1f} ms")
        for d in result.detections:
            print(f"    {d.class_name:15s} score={d.score:.2f}  bbox={[f'{v:.0f}' for v in d.bbox]}")
        print()
