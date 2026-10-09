#!/usr/bin/env python3
"""Experiment B: probability calibration x decision calibration (2x2).

Two levers are usually conflated:

  probability calibration : raw forecast p vs temperature-scaled p  (ECE lever)
  decision calibration    : fixed threshold vs DRRL-VoI regime threshold (utility lever)

Because the standard calibrators (temperature, Platt, isotonic) are *monotone*,
they cannot change the set of actions a threshold rule can express. To make the
comparison decision-fair we map the threshold grid through the same monotone map,
so the calibrated cell evaluates the identical decision set on a rescaled axis.

Prediction of the Regime-Aware Decision Calibration thesis:

  * the ECE lever (probability calibration) is decision-inert -> ~0 utility effect;
  * the decision lever (regime-aware threshold) moves utility;
  * a naive fixed 0.5 anchor in probability units only *appears* to make calibration
    matter, which is an artifact of the arbitrary anchor, not economics.

Probability calibration is fit on the validation split only. Decision layers are
fit on search+validation and evaluated on external. No new assets are used.
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

from src.rutl.data import generate_synthetic_scenario
from src.rutl.decision_regret import brier_score, expected_calibration_error
from src.rutl.drrl import evaluate_drrl
from src.rutl.group_metrics import grouped_metrics, summarize_group_metrics

RAW_GRID = np.linspace(0.30, 0.80, 51)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling by deterministic grid search on validation NLL."""
    z = _logit(p)
    y = np.asarray(y, dtype=float)
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.25, 4.0, 76):
        q = _sigmoid(z / t)
        nll = -float(np.mean(y * np.log(q + 1e-12) + (1.0 - y) * np.log(1.0 - q + 1e-12)))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    return _sigmoid(_logit(p) / temperature)


def _fixed_cell(validation: pd.DataFrame, external: pd.DataFrame, *, bootstrap: dict, grid: np.ndarray, fixed_threshold: float | None = None) -> dict:
    if fixed_threshold is None:
        best_t, best = float(grid[0]), -1e9
        for threshold in grid:
            actions = (validation["p"].to_numpy(dtype=float) >= float(threshold)).astype(float)
            summary = summarize_group_metrics(
                grouped_metrics(validation, actions, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
            )
            if summary["mean_nu_lb"] > best:
                best, best_t = summary["mean_nu_lb"], float(threshold)
    else:
        best_t = float(fixed_threshold)
    actions = (external["p"].to_numpy(dtype=float) >= best_t).astype(float)
    summary = summarize_group_metrics(
        grouped_metrics(external, actions, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
    )
    return {"threshold": best_t, "activated": True, "mean_nu_lb": summary["mean_nu_lb"]}


def _with_score(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    out["p"] = np.asarray(score, dtype=float)
    return out


def _cells_for_split(search, validation, external, *, protocol, drrl_config) -> dict:
    bootstrap = protocol["primary_metric"]["bootstrap"]
    temperature = fit_temperature(validation["p"].to_numpy(float), validation["y"].to_numpy(float))
    calibrated_grid = apply_temperature(RAW_GRID, temperature)

    raw = (search, validation, external)
    cal = (
        _with_score(search, apply_temperature(search["p"].to_numpy(float), temperature)),
        _with_score(validation, apply_temperature(validation["p"].to_numpy(float), temperature)),
        _with_score(external, apply_temperature(external["p"].to_numpy(float), temperature)),
    )
    cells: dict = {}
    for prob_name, (s, v, e), grid in [("raw", raw, RAW_GRID), ("calibrated", cal, calibrated_grid)]:
        ece = expected_calibration_error(e["p"].to_numpy(float), e["y"].to_numpy(float))
        brier = brier_score(e["p"].to_numpy(float), e["y"].to_numpy(float))
        fixed = _fixed_cell(v, e, bootstrap=bootstrap, grid=grid)
        drrl = evaluate_drrl(s, v, e, protocol=protocol, config=drrl_config, grid=grid)
        naive = _fixed_cell(v, e, bootstrap=bootstrap, grid=grid, fixed_threshold=0.5)
        cells[(prob_name, "fixed")] = {"ece": ece, "brier": brier, "temperature": temperature, **fixed}
        cells[(prob_name, "drrl")] = {
            "ece": ece, "brier": brier, "temperature": temperature,
            "threshold": drrl["fixed_threshold"], "activated": drrl["activated"],
            "mean_nu_lb": drrl["external_policy"]["mean_nu_lb"],
        }
        cells[(prob_name, "fixed_0.50")] = {"ece": ece, "brier": brier, "temperature": temperature, **naive}
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "data_splits.json"))
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "evaluation_protocol.json"))
    parser.add_argument("--bundle", default=str(ROOT / "artifacts" / "bundles" / "predictions_v1.csv"))
    parser.add_argument("--drrl-config", default=str(ROOT / "direction_evolve2" / "experiment_b_drrl_config.json"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--outdir", default=str(ROOT / "artifacts" / "results"))
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    drrl_config = json.loads(Path(args.drrl_config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.config).read_text(encoding="utf-8"))
    scenarios = {s["name"]: s for s in manifest["synthetic_scenarios"]}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for name in ["decision_relevant_state", "asymmetric_utility", "high_separation", "uninformative_regime"]:
        base = scenarios[name]
        for offset in range(args.repeats):
            scenario = dict(base)
            scenario["seed"] = int(base["seed"]) + offset * 101
            frame = generate_synthetic_scenario(scenario)
            s = frame[frame["split"] == "search"].reset_index(drop=True)
            v = frame[frame["split"] == "validation"].reset_index(drop=True)
            e = frame[frame["split"] == "external"].reset_index(drop=True)
            cells = _cells_for_split(s, v, e, protocol=protocol, drrl_config=drrl_config)
            base_nu = cells[("raw", "fixed")]["mean_nu_lb"]
            for (prob, decision), cell in cells.items():
                rows.append({
                    "dataset": "synthetic", "scenario": name, "seed": scenario["seed"],
                    "prob_cal": prob, "decision": decision, "temperature": cell["temperature"],
                    "ece": cell["ece"], "brier": cell["brier"], "activated": cell["activated"],
                    "mean_nu_lb": cell["mean_nu_lb"], "delta_vs_raw_fixed": cell["mean_nu_lb"] - base_nu,
                })

    # Real: single global monotone temperature on the pooled BTC validation split.
    bundle = pd.read_csv(args.bundle, parse_dates=["date"], low_memory=False)
    real = bundle[bundle["asset"].isin(["BTC-USD", "TSLA"])].copy()
    btc = real[real["asset"] == "BTC-USD"]
    search = btc[btc["split"] == "search"].reset_index(drop=True)
    validation = btc[btc["split"] == "validation"].reset_index(drop=True)
    external = real[real["split"] == "external"].reset_index(drop=True)
    cells = _cells_for_split(search, validation, external, protocol=protocol, drrl_config=drrl_config)
    base_nu = cells[("raw", "fixed")]["mean_nu_lb"]
    for (prob, decision), cell in cells.items():
        rows.append({
            "dataset": "real", "scenario": "BTC+TSLA", "seed": 20260928,
            "prob_cal": prob, "decision": decision, "temperature": cell["temperature"],
            "ece": cell["ece"], "brier": cell["brier"], "activated": cell["activated"],
            "mean_nu_lb": cell["mean_nu_lb"], "delta_vs_raw_fixed": cell["mean_nu_lb"] - base_nu,
        })

    result = pd.DataFrame(rows)
    result.to_csv(outdir / "calibration_decision_2x2.csv", index=False)

    def _agg(dataset: str) -> dict:
        sub = result[result["dataset"] == dataset]
        entry = {}
        for prob in ["raw", "calibrated"]:
            for decision in ["fixed", "drrl", "fixed_0.50"]:
                cell = sub[(sub.prob_cal == prob) & (sub.decision == decision)]
                entry[f"{prob}_{decision}"] = {
                    "mean_nu_lb": float(cell["mean_nu_lb"].mean()) if len(cell) else float("nan"),
                    "ece": float(cell["ece"].mean()) if len(cell) else float("nan"),
                    "activation_rate": float(cell["activated"].mean()) if len(cell) else float("nan"),
                }
        # Decision-fair (monotone-mapped grid) factorization effects.
        entry["effect_prob_cal_within_fixed"] = entry["calibrated_fixed"]["mean_nu_lb"] - entry["raw_fixed"]["mean_nu_lb"]
        entry["effect_decision_cal_within_raw"] = entry["raw_drrl"]["mean_nu_lb"] - entry["raw_fixed"]["mean_nu_lb"]
        entry["effect_decision_cal_within_calibrated"] = entry["calibrated_drrl"]["mean_nu_lb"] - entry["calibrated_fixed"]["mean_nu_lb"]
        entry["interaction"] = entry["effect_decision_cal_within_calibrated"] - entry["effect_decision_cal_within_raw"]
        # Naive 0.5-anchor artifact.
        entry["effect_prob_cal_at_naive_0.50"] = entry["calibrated_fixed_0.50"]["mean_nu_lb"] - entry["raw_fixed_0.50"]["mean_nu_lb"]
        entry["ece_raw"] = entry["raw_fixed"]["ece"]
        entry["ece_calibrated"] = entry["calibrated_fixed"]["ece"]
        return entry

    summary = {"by_dataset": {"synthetic": _agg("synthetic"), "real": _agg("real")}}
    summary["by_scenario"] = {}
    for scenario, sub in result[result["dataset"] == "synthetic"].groupby("scenario"):
        summary["by_scenario"][scenario] = {
            f"{p}_{d}": float(sub[(sub.prob_cal == p) & (sub.decision == d)]["mean_nu_lb"].mean())
            for p in ["raw", "calibrated"] for d in ["fixed", "drrl"]
        }
        summary["by_scenario"][scenario]["ece_raw"] = float(sub[(sub.prob_cal == "raw")]["ece"].mean())
        summary["by_scenario"][scenario]["ece_calibrated"] = float(sub[(sub.prob_cal == "calibrated")]["ece"].mean())
        summary["by_scenario"][scenario]["drrl_activation"] = float(sub[(sub.prob_cal == "calibrated") & (sub.decision == "drrl")]["activated"].mean())
    (outdir / "calibration_decision_2x2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(result.groupby(["dataset", "scenario", "prob_cal", "decision"]).agg(
        ece=("ece", "mean"), mean_nu_lb=("mean_nu_lb", "mean"), activated=("activated", "mean")
    ).to_string())
    print("\nFactorization effects:")
    for dataset in ["synthetic", "real"]:
        print(f"[{dataset}]")
        for k, v in summary["by_dataset"][dataset].items():
            if isinstance(v, float):
                print(f"  {k} = {v:+.4f}")
        print(f"  ECE: {summary['by_dataset'][dataset]['ece_raw']:.4f} -> {summary['by_dataset'][dataset]['ece_calibrated']:.4f}")


if __name__ == "__main__":
    main()
