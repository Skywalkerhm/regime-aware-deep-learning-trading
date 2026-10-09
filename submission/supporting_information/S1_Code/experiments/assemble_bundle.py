#!/usr/bin/env python3
"""Assemble synthetic and real predictions into the canonical RUTL bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", default=str(ROOT / "artifacts" / "bundles" / "synthetic_v1.csv"))
    parser.add_argument("--real", default=str(ROOT / "artifacts" / "bundles" / "real_v1.csv"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "bundles" / "predictions_v1.csv"))
    args = parser.parse_args()
    frames = [pd.read_csv(args.synthetic), pd.read_csv(args.real)]
    bundle = pd.concat(frames, ignore_index=True, sort=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle.to_csv(output, index=False)
    print(json.dumps({
        "output": str(output),
        "rows": len(bundle),
        "assets": sorted(bundle["asset"].unique().tolist()),
        "splits": bundle["split"].value_counts().to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
