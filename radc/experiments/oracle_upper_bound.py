#!/usr/bin/env python3
"""Experiment C: oracle upper bound and DRRL recovery under RADC.

For the decision-calibration mechanism we measure, on the external split:

  fixed_nu_lb        : validation-selected fixed threshold (applied protocol)
  oracle_lb_nu_lb    : per-true-state thresholds selected on search (deployable
                       upper bound -- true state known, thresholds not look-ahead)
  oracle_ub_nu_lb    : per-true-state thresholds selected on external (optimistic
                       in-sample ceiling)
  drrl_nu_lb         : DRRL-VoI regime-aware threshold

Derived quantities:
  oracle_lb_gain     = oracle_lb_nu_lb - fixed_nu_lb
  recovery           = drrl_delta / oracle_lb_gain   (fraction of the bound captured)
  abstain_near_zero  = DRRL does not activate when oracle_lb_gain is near zero

Real BTC/TSLA has no true states: the detector posterior argmax is used as an
explicitly labelled *pseudo-oracle proxy*, never as a true oracle. No new assets.
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
from src.rutl.decision_regret import detector_states, true_states
from src.rutl.drrl import evaluate_drrl
from src.rutl.group_metrics import grouped_metrics, select_fixed_threshold_grouped, summarize_group_metrics

NU_GRID = np.linspace(0.30, 0.80, 51)
NEAR_ZERO = 0.05  # oracle gain below this is "no meaningful decision-relevant state"


def _best_state_thresholds(part: pd.DataFrame, state_col: str) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for state in sorted(part[state_col].unique()):
        block = part[part[state_col] == state]
        best_t, best = float(NU_GRID[0]), -1e9
        for threshold in NU_GRID:
            actions = (block["p"].to_numpy(float) >= float(threshold)).astype(float)
            result = evaluate_actions(actions, block["r_next"].to_numpy(float), transaction_cost=0.001, drawdown_penalty=0.25)
            score = result.annualized_mean - 0.05 * result.annualized_vol - 0.10 * result.max_drawdown
            if score > best:
                best, best_t = score, float(threshold)
        thresholds[int(state)] = best_t
    return thresholds


def _state_actions(frame: pd.DataFrame, thresholds: dict[int, float], state_col: str) -> np.ndarray:
    states = frame[state_col].to_numpy(dtype=int)
    default = float(np.mean(list(thresholds.values())))
    cutoffs = np.array([thresholds.get(int(s), default) for s in states], dtype=float)
    return (frame["p"].to_numpy(float) >= cutoffs).astype(float)


def _nu_lb(frame: pd.DataFrame, actions: np.ndarray, bootstrap: dict) -> float:
    return summarize_group_metrics(
        grouped_metrics(frame, actions, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
    )["mean_nu_lb"]


def _scenario_row(search, validation, external, *, state_col, protocol, drrl_config) -> dict:
    bootstrap = protocol["primary_metric"]["bootstrap"]
    fixed_threshold, _, _ = select_fixed_threshold_grouped(
        validation, grid=NU_GRID, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap, score_column="p"
    )
    fixed_actions = (external["p"].to_numpy(float) >= float(fixed_threshold)).astype(float)
    fixed_nu = _nu_lb(external, fixed_actions, bootstrap)

    oracle_lb_actions = _state_actions(external, _best_state_thresholds(search, state_col), state_col)
    oracle_ub_actions = _state_actions(external, _best_state_thresholds(external, state_col), state_col)

    drrl = evaluate_drrl(search, validation, external, protocol=protocol, config=drrl_config)
    oracle_lb_gain = _nu_lb(external, oracle_lb_actions, bootstrap) - fixed_nu
    drrl_delta = drrl["external_delta"]
    return {
        "fixed_nu_lb": fixed_nu,
        "oracle_lb_nu_lb": _nu_lb(external, oracle_lb_actions, bootstrap),
        "oracle_ub_nu_lb": _nu_lb(external, oracle_ub_actions, bootstrap),
        "oracle_lb_gain": oracle_lb_gain,
        "oracle_ub_gain": _nu_lb(external, oracle_ub_actions, bootstrap) - fixed_nu,
        "drrl_delta": drrl_delta,
        "drrl_activated": bool(drrl["activated"]),
        "recovery": drrl_delta / oracle_lb_gain if oracle_lb_gain > 1e-9 else np.nan,
        "near_zero_oracle": bool(abs(oracle_lb_gain) < NEAR_ZERO),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "data_splits.json"))
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "evaluation_protocol.json"))
    parser.add_argument("--bundle", default=str(ROOT / "artifacts" / "bundles" / "predictions_v1.csv"))
    parser.add_argument("--drrl-config", default=str(ROOT / "direction_evolve2" / "experiment_b_drrl_config.json"))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--outdir", default=str(ROOT / "artifacts" / "results"))
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    drrl_config = json.loads(Path(args.drrl_config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.config).read_text(encoding="utf-8"))
    scenarios = {s["name"]: s for s in manifest["synthetic_scenarios"]}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for name, base in scenarios.items():
        for offset in range(args.repeats):
            scenario = dict(base)
            scenario["seed"] = int(base["seed"]) + offset * 101
            frame = generate_synthetic_scenario(scenario)
            frame["state_true"] = true_states(frame)
            s = frame[frame["split"] == "search"].reset_index(drop=True)
            v = frame[frame["split"] == "validation"].reset_index(drop=True)
            e = frame[frame["split"] == "external"].reset_index(drop=True)
            row = _scenario_row(s, v, e, state_col="state_true", protocol=protocol, drrl_config=drrl_config)
            rows.append({"dataset": "synthetic", "scenario": name, "seed": scenario["seed"], **row})

    # Real: BTC search/validation train; detector posterior as labelled pseudo-oracle.
    bundle = pd.read_csv(args.bundle, parse_dates=["date"], low_memory=False)
    real = bundle[bundle["asset"].isin(["BTC-USD", "TSLA"])].copy()
    real["state_det"] = detector_states(real)
    btc = real[real["asset"] == "BTC-USD"]
    s = btc[btc["split"] == "search"].reset_index(drop=True)
    v = btc[btc["split"] == "validation"].reset_index(drop=True)
    e = real[real["split"] == "external"].reset_index(drop=True)
    row = _scenario_row(s, v, e, state_col="state_det", protocol=protocol, drrl_config=drrl_config)
    rows.append({"dataset": "real_pseudo_oracle", "scenario": "BTC_train_TSLA+BTC_external", "seed": 20260928, **row})

    result = pd.DataFrame(rows)
    result.to_csv(outdir / "oracle_upper_bound.csv", index=False)

    summary = {"by_scenario": [], "overall": {}}
    for name, sub in result[result["dataset"] == "synthetic"].groupby("scenario"):
        near_zero = sub[sub["near_zero_oracle"]]
        summary["by_scenario"].append({
            "scenario": name,
            "runs": int(len(sub)),
            "mean_oracle_lb_gain": float(sub["oracle_lb_gain"].mean()),
            "mean_oracle_ub_gain": float(sub["oracle_ub_gain"].mean()),
            "mean_drrl_delta": float(sub["drrl_delta"].mean()),
            "mean_recovery": float(sub["recovery"].mean()),
            "activation_rate": float(sub["drrl_activated"].mean()),
            "abstain_rate_when_near_zero": float((~near_zero["drrl_activated"]).mean()) if len(near_zero) else float("nan"),
        })
    syn = result[result["dataset"] == "synthetic"]
    summary["overall"] = {
        "corr_oracle_lb_gain__drrl_delta": float(np.corrcoef(syn["oracle_lb_gain"], syn["drrl_delta"])[0, 1]) if syn["oracle_lb_gain"].std() > 1e-12 else float("nan"),
        "mean_recovery_positive_controls": float(syn[syn["oracle_lb_gain"] > NEAR_ZERO]["recovery"].mean()),
        "activation_rate_positive_controls": float(syn[syn["oracle_lb_gain"] > NEAR_ZERO]["drrl_activated"].mean()),
        "activation_rate_negative_controls": float(syn[syn["oracle_lb_gain"] <= NEAR_ZERO]["drrl_activated"].mean()),
    }
    (outdir / "oracle_upper_bound_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(pd.DataFrame(summary["by_scenario"]).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\nOverall:")
    for k, v in summary["overall"].items():
        print(f"  {k} = {v:+.4f}")
    print("\nReal (pseudo-oracle proxy):")
    print(result[result["dataset"] == "real_pseudo_oracle"][[
        "fixed_nu_lb", "oracle_lb_nu_lb", "oracle_lb_gain", "drrl_delta", "drrl_activated", "recovery"
    ]].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
