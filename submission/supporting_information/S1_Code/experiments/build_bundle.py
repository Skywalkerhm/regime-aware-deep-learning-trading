#!/usr/bin/env python3
"""Build the synthetic RUTL v1 prediction bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.rutl.data import generate_synthetic_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "data_splits.json"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "bundles" / "synthetic_v1.csv"))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    frames = [generate_synthetic_scenario(s) for s in config["synthetic_scenarios"]]
    bundle = pd.concat(frames, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle.to_csv(output, index=False)
    print(json.dumps({
        "output": str(output),
        "rows": len(bundle),
        "scenarios": bundle["asset"].nunique(),
        "splits": bundle["split"].value_counts().to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
