#!/usr/bin/env python3
"""Evaluate one RUTL candidate and print the AlphaEvolve JSON schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rutl.evaluator import evaluate_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--bundle", default=str(ROOT / "artifacts" / "bundles" / "predictions_v1.csv"))
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "evaluation_protocol.json"))
    parser.add_argument("--include-external", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    result = evaluate_candidate(
        Path(args.candidate),
        Path(args.bundle),
        Path(args.protocol),
        include_external=args.include_external,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
