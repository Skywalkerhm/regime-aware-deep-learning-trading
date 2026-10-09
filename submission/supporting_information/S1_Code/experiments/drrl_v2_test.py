#!/usr/bin/env python3
"""Test DRRL v0.2 depth-2 tree on controlled and real data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rutl.data import generate_synthetic_scenario
from src.rutl.drrl_v2 import evaluate_drrl_v2
from src.rutl.evaluator import _frame_for_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "results" / "drrl_v2_test.json"))
    args = parser.parse_args()
    protocol = json.loads((ROOT / "configs" / "evaluation_protocol.json").read_text(encoding="utf-8"))
    config = {
        "max_depth": 2,
        "features": ["uncertainty", "volatility_20d", "trend_strength", "posterior_crisis"],
        "quantiles": [0.25, 0.5, 0.75],
        "complexity_penalty": 0.01,
        "activation_margin": 0.05,
        "min_positive_fraction": 0.9,
        "vol_penalty": 0.05,
        "dd_penalty": 0.10,
        "dd_tolerance": 0.10,
        "turnover_tolerance": 0.25,
        "grid_size": 31,
        "min_leaf": 80,
    }
    manifest = json.loads((ROOT / "configs" / "data_splits.json").read_text(encoding="utf-8"))
    scenarios = {s["name"]: s for s in manifest["synthetic_scenarios"]}
    rows = []
    for name in ["decision_relevant_state", "asymmetric_utility", "uninformative_regime", "weak_detector"]:
        base = scenarios[name]
        for offset in range(args.repeats):
            scenario = dict(base)
            scenario["seed"] = int(base["seed"]) + offset * 101
            frame = generate_synthetic_scenario(scenario)
            result = evaluate_drrl_v2(
                _frame_for_split(frame, "search"),
                _frame_for_split(frame, "validation"),
                _frame_for_split(frame, "external"),
                protocol=protocol,
                config=config,
            )
            rows.append({"scenario": name, "seed": scenario["seed"], **result})
    bundle = pd.read_csv(ROOT / "artifacts" / "bundles" / "predictions_v1.csv", parse_dates=["date"], low_memory=False)
    real = bundle[bundle["asset"].isin(["BTC-USD", "TSLA"])]
    real_result = evaluate_drrl_v2(
        _frame_for_split(real[real["asset"] == "BTC-USD"], "search"),
        _frame_for_split(real[real["asset"] == "BTC-USD"], "validation"),
        _frame_for_split(real, "external"),
        protocol=protocol,
        config=config,
    )
    rows.append({"scenario": "real", "seed": 20260928, **real_result})
    output = Path(args.output)
    output.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    summary = pd.DataFrame(rows).groupby("scenario").agg(
        runs=("seed", "count"),
        activation_rate=("activated", "mean"),
        mean_delta=("external_delta", "mean"),
        mean_splits=("n_splits", "mean"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
