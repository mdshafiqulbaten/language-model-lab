from __future__ import annotations

import argparse
import json
from pathlib import Path


PRESETS = {
    "local_cpu": {"layers": 4, "width": 128, "vocab": 320, "context": 128},
    "consumer_gpu": {"layers": 8, "width": 512, "vocab": 32000, "context": 1024},
    "cloud_gpu": {"layers": 24, "width": 2048, "vocab": 50000, "context": 4096},
}


def estimate(preset: str) -> dict:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset: {preset}")
    cfg = PRESETS[preset]
    width, layers, vocab = cfg["width"], cfg["layers"], cfg["vocab"]
    parameters = vocab * width + layers * (12 * width * width + 8 * width)
    return {
        "preset": preset,
        "configuration": cfg,
        "estimated_parameters": parameters,
        "fp32_weight_bytes": parameters * 4,
        "adam_training_bytes_lower_bound": parameters * 16,
        "warning": "Measure a pilot; activations, cache, kernels, and fragmentation are excluded.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="local_cpu")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args(argv)
    report = estimate(args.preset)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

