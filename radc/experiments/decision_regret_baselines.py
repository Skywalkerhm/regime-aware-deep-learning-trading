#!/usr/bin/env python3
"""Baseline comparison in decision-regret and realized-utility units.

For the same external split, compares:

  1. fixed        : validation-selected fixed threshold (group NU_LB);
  2. rule         : current rule-based regime thresholds (crisis=0.80, ...);
  3. tuned_regime : per-detector-state thresholds tuned on validation NU_LB;
  4. oracle_state : per-true-state thresholds selected on search (synthetic only).

Reports both the mean decision regret (relative to a common conditional Delta
reference) and the realized external group NU_LB. No new assets are used.
"""

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
from src.rutl.decision_regret import (
    RULE_LABELS,
    RULE_THRESHOLDS,
    _bin_index,
    build_delta_table,
    decision_regret,
    default_score_edges,
    detector_labels,
    detector_states,
    true_states,
)
from src.rutl.group_metrics import grouped_metrics, select_fixed_threshold_grouped, summarize_group_metrics

NU_GRID = np.linspace(0.30, 0.80, 51)
EDGES = default_score_edges(10)


def _group_nu_lb(frame: pd.DataFrame, actions: np.ndarray, bootstrap: dict) -> float:
    return summarize_group_metrics(
        grouped_metrics(frame, actions, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
    )["mean_nu_lb"]


def _best_threshold_nu(part: pd.DataFrame, bootstrap: dict) -> float:
    best_t, best = float(NU_GRID[0]), -1e9
    for threshold in NU_GRID:
        actions = (part["p"].to_numpy(dtype=float) >= float(threshold)).astype(float)
        value = _group_nu_lb(part, actions, bootstrap)
        if value > best:
            best, best_t = value, float(threshold)
    return best_t


def _best_state_thresholds_nu(part: pd.DataFrame, state_column: str, bootstrap: dict) -> dict[int, float]:
    out: dict[int, float] = {}
    for state in sorted(part[state_column].unique()):
        out[int(state)] = _best_threshold_nu(part[part[state_column] == state], bootstrap)
    return out


def _actions_from_map(frame: pd.DataFrame, thresholds: dict[int, float], states: np.ndarray) -> np.ndarray:
    default = float(np.mean(list(thresholds.values())))
    p = frame["p"].to_numpy(dtype=float)
    cutoffs = np.array([thresholds.get(int(s), default) for s in states], dtype=float)
    return (p >= cutoffs).astype(float)


def _regret(frame: pd.DataFrame, actions: np.ndarray, states: np.ndarray, table) -> float:
    bins = _bin_index(frame["p"].to_numpy(dtype=float), EDGES)
    return decision_regret(bins, states, actions, table)


def _evaluate_scope(
    scope: str,
    validation: pd.DataFrame,
    search: pd.DataFrame,
    external: pd.DataFrame,
    *,
    bootstrap: dict,
    reference_state_column: str,
    include_oracle: bool,
) -> list[dict]:
    fixed_threshold, _, _ = select_fixed_threshold_grouped(
        validation, grid=NU_GRID, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap, score_column="p"
    )
    states_ref = external[reference_state_column].to_numpy(dtype=int)
    table = build_delta_table(external, states_ref, edges=EDGES)

    det_states = detector_states(external)
    labels = detector_labels(external)
    rule_actions = np.array([RULE_THRESHOLDS[label] for label in labels], dtype=float)
    rule_actions = (external["p"].to_numpy(dtype=float) >= rule_actions).astype(float)

    tuned = _best_state_thresholds_nu(validation, "state_det", bootstrap)
    tuned_actions = _actions_from_map(external, tuned, det_states)

    fixed_actions = (external["p"].to_numpy(dtype=float) >= float(fixed_threshold)).astype(float)

    policies = {
        "fixed": fixed_actions,
        "rule": rule_actions,
        "tuned_regime": tuned_actions,
    }
    if include_oracle:
        oracle = _best_state_thresholds_nu(search, "state_true", bootstrap)
        policies["oracle_state"] = _actions_from_map(external, oracle, true_states(external))

    fixed_nu = _group_nu_lb(external, fixed_actions, bootstrap)
    rows = []
    for name, actions in policies.items():
        rows.append({
            "scope": scope,
            "policy": name,
            "regret": _regret(external, actions, states_ref, table),
            "mean_nu_lb": _group_nu_lb(external, actions, bootstrap),
            "delta_nu_lb_vs_fixed": _group_nu_lb(external, actions, bootstrap) - fixed_nu,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "data_splits.json"))
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "evaluation_protocol.json"))
    parser.add_argument("--bundle", default=str(ROOT / "artifacts" / "bundles" / "predictions_v1.csv"))
    parser.add_argument("--outdir", default=str(ROOT / "artifacts" / "results"))
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    bootstrap = protocol["primary_metric"]["bootstrap"]
    outdir = Path(args.outdir)

    rows: list[dict] = []

    # Synthetic controls (true state reference, includes oracle row).
    manifest = json.loads(Path(args.config).read_text(encoding="utf-8"))
    scenarios = {s["name"]: s for s in manifest["synthetic_scenarios"]}
    for name in ["decision_relevant_state", "asymmetric_utility", "high_separation", "uninformative_regime"]:
        base = scenarios[name]
        frame = generate_synthetic_scenario(base)
        frame["state_true"] = true_states(frame)
        frame["state_det"] = detector_states(frame)
        search = frame[frame["split"] == "search"].reset_index(drop=True)
        validation = frame[frame["split"] == "validation"].reset_index(drop=True)
        external = frame[frame["split"] == "external"].reset_index(drop=True)
        rows += _evaluate_scope(
            f"synthetic:{name}", validation, search, external,
            bootstrap=bootstrap, reference_state_column="state_true", include_oracle=True,
        )

    # Real: train thresholds on BTC development folds; detector reference proxy.
    bundle = pd.read_csv(args.bundle, parse_dates=["date"], low_memory=False)
    real = bundle[bundle["asset"].isin(["BTC-USD", "TSLA"])].copy()
    real["state_det"] = detector_states(real)
    btc = real[real["asset"] == "BTC-USD"]
    btc_search = btc[btc["split"] == "search"].reset_index(drop=True)
    btc_validation = btc[btc["split"] == "validation"].reset_index(drop=True)
    external_all = real[real["split"] == "external"].reset_index(drop=True)
    for label, part in [("real:ALL", external_all), ("real:BTC-USD", external_all[external_all["asset"] == "BTC-USD"].reset_index(drop=True)),
                        ("real:TSLA", external_all[external_all["asset"] == "TSLA"].reset_index(drop=True))]:
        rows += _evaluate_scope(
            label, btc_validation, btc_search, part,
            bootstrap=bootstrap, reference_state_column="state_det", include_oracle=False,
        )

    result = pd.DataFrame(rows)
    result.to_csv(outdir / "decision_regret_baselines.csv", index=False)
    pd.set_option("display.width", 220)
    print(result.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))


if __name__ == "__main__":
    main()
