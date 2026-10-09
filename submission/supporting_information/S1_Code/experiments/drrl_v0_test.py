#!/usr/bin/env python3
"""Run DRRL-VoI v0.1 on synthetic controls and real BTC/TSLA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rutl.drrl import evaluate_drrl
from src.rutl.evaluator import _frame_for_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(ROOT / "artifacts" / "bundles" / "predictions_v1.csv"))
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "evaluation_protocol.json"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "results" / "drrl_v0_test.json"))
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    bundle = pd.read_csv(args.bundle, parse_dates=["date"], low_memory=False)
    synthetic = bundle[bundle["model"] == "synthetic"].copy()
    real = bundle[bundle["asset"].isin(["BTC-USD", "TSLA"])].copy()
    results = []

    for scenario in sorted(synthetic["asset"].unique()):
        part = synthetic[synthetic["asset"] == scenario]
        result = evaluate_drrl(
            _frame_for_split(part, "search"),
            _frame_for_split(part, "validation"),
            _frame_for_split(part, "external"),
            protocol=protocol,
        )
        results.append({"dataset": "synthetic", "name": scenario, **result})

    real_search = _frame_for_split(real, "search")
    real_validation = _frame_for_split(real, "validation")
    real_external = _frame_for_split(real, "external")
    real_search = real_search[real_search["asset"] == "BTC-USD"]
    real_validation = real_validation[real_validation["asset"] == "BTC-USD"]
    real_external = real_external[real_external["asset"].isin(["BTC-USD", "TSLA"])]
    result = evaluate_drrl(real_search, real_validation, real_external, protocol=protocol)
    results.append({"dataset": "real", "name": "BTC_search_TSLA_external", **result})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
