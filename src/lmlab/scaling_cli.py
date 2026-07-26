from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    runs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    normalized = []
    for run in runs:
        required = {"name", "parameters", "tokens", "validation_loss"}
        if not required <= set(run):
            raise ValueError(f"run is missing fields: {required - set(run)}")
        normalized.append(
            {
                **run,
                "estimated_training_flops": 6 * run["parameters"] * run["tokens"],
            }
        )
    report = {"run_count": len(normalized), "runs": normalized}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

