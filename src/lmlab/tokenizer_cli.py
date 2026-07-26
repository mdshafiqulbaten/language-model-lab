from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .tokenizer import ByteBPETokenizer


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--input", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--report", required=True)
    train.add_argument("--vocab-size", type=int, default=320)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--tokenizer", required=True)
    evaluate.add_argument("--samples", required=True)
    evaluate.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    if args.command == "train":
        texts = Path(args.input).read_text(encoding="utf-8").splitlines()
        tokenizer = ByteBPETokenizer.train(texts, args.vocab_size)
        tokenizer.save(args.output)
        report = {"samples": len(texts), "learned_merges": len(tokenizer.merges)}
    else:
        tokenizer = ByteBPETokenizer.load(args.tokenizer)
        rows = []
        with open(args.samples, encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if not row:
                    continue
                language, text = row[0], row[-1]
                ids = tokenizer.encode(text)
                rows.append(
                    {
                        "language": language,
                        "text": text,
                        "tokens": len(ids),
                        "round_trip": tokenizer.decode(ids) == text,
                    }
                )
        report = {"sample_count": len(rows), "results": rows}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

