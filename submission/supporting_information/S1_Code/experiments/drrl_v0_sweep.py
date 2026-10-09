#!/usr/bin/env python3
"""Multi-seed test of DRRL-VoI with oracle upper bounds on synthetic DGPs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rutl.backtest import evaluate_actions
from src.rutl.data import generate_synthetic_scenario
from src.rutl.drrl import evaluate_drrl
from src.rutl.evaluator import _frame_for_split
from src.rutl.group_metrics import grouped_metrics, summarize_group_metrics


def _best_regime_thresholds(search: pd.DataFrame, grid: np.ndarray) -> dict[int, float]:
    thresholds = {}
    for state in sorted(search["truth_regime"].unique()):
        part = search[search["truth_regime"] == state]
        best_threshold = float(grid[0])
        best_score = -1e9
        for threshold in grid:
            actions = (part["p"].to_numpy(dtype=float) >= float(threshold)).astype(float)
            result = evaluate_actions(actions, part["r_next"].to_numpy(dtype=float), transaction_cost=0.001, drawdown_penalty=0.25)
            score = result.annualized_mean - 0.05 * result.annualized_vol - 0.10 * result.max_drawdown
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds[int(state)] = best_threshold
    return thresholds


def _oracle_actions(frame: pd.DataFrame, thresholds: dict[int, float]) -> np.ndarray:
    states = frame["truth_regime"].to_numpy(dtype=int)
    cutoffs = np.array([thresholds[int(s)] for s in states], dtype=float)
    return (frame["p"].to_numpy(dtype=float) >= cutoffs).astype(float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "data_splits.json"))
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "evaluation_protocol.json"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "results" / "drrl_v0_sweep.json"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--drrl-config", default=None)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    drrl_config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else None
    bootstrap = protocol["primary_metric"]["bootstrap"]
    grid = np.linspace(0.30, 0.80, 51)
    rows = []

    for base in config["synthetic_scenarios"]:
        if args.scenario and base["name"] not in args.scenario:
            continue
        for seed_offset in range(args.repeats):
            scenario = dict(base)
            scenario["seed"] = int(base["seed"]) + seed_offset * 101
            frame = generate_synthetic_scenario(scenario)
            search = _frame_for_split(frame, "search")
            validation = _frame_for_split(frame, "validation")
            external = _frame_for_split(frame, "external")
            drrl = evaluate_drrl(search, validation, external, protocol=protocol, config=drrl_config)
            thresholds = _best_regime_thresholds(search, grid)
            oracle_metrics = summarize_group_metrics(
                grouped_metrics(
                    external,
                    _oracle_actions(external, thresholds),
                    transaction_cost=0.001,
                    drawdown_penalty=0.25,
                    bootstrap=bootstrap,
                )
            )
            oracle_delta = oracle_metrics["mean_nu_lb"] - drrl["external_fixed"]["mean_nu_lb"]
            rows.append({
                "scenario": base["name"],
                "seed": scenario["seed"],
                "activated": drrl["activated"],
                "voi_low": drrl["voi_low"],
                "voi_high": drrl["voi_high"],
                "drrl_delta": drrl["external_delta"],
                "oracle_delta": oracle_delta,
                "recovery_ratio": drrl["external_delta"] / oracle_delta if oracle_delta > 1e-9 else np.nan,
                "split_feature": None if drrl["split_policy"] is None else drrl["split_policy"]["feature"],
                "fixed_threshold": drrl["fixed_threshold"],
            })
    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output.with_suffix(".csv"), index=False)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = result.groupby("scenario").agg(
        runs=("seed", "count"),
        activation_rate=("activated", "mean"),
        mean_drrl_delta=("drrl_delta", "mean"),
        mean_oracle_delta=("oracle_delta", "mean"),
        mean_recovery_ratio=("recovery_ratio", "mean"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
