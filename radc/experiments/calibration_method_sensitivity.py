#!/usr/bin/env python3
"""Sensitivity of the decision-inertness result (Proposition 4) to the calibrator.

Experiment B used temperature scaling. Here we repeat the decision-fair comparison
for temperature, Platt, and isotonic calibration on the synthetic controls and the
real BTC/TSLA bundle. For each monotone calibrator we map the raw threshold grid
through it and check whether the probability-calibration effect on external NU_LB is
zero, as the invariance proposition predicts. Isotonic regression is monotone but not
strictly monotone, so it is the informative stress case. No new assets.
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

from experiments.calibration_decision_2x2 import apply_temperature, fit_temperature
from src.rutl.data import generate_synthetic_scenario
from src.rutl.decision_regret import detector_states, true_states
from src.rutl.group_metrics import grouped_metrics, summarize_group_metrics

RAW_GRID = np.linspace(0.30, 0.80, 51)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def fit_platt(p, y):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(_logit(p).reshape(-1, 1), y)
    return model


def apply_platt(p, model):
    return model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def fit_isotonic(p, y):
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(out_of_bounds="clip", increasing=True)
    model.fit(p, y)
    return model


def apply_isotonic(p, model):
    return model.transform(p)


def _fixed_nu(validation, external, score_col, grid, bootstrap):
    best_t, best = float(grid[0]), -1e9
    for t in grid:
        a = (validation[score_col].to_numpy(float) >= float(t)).astype(float)
        s = summarize_group_metrics(grouped_metrics(validation, a, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap))
        if s["mean_nu_lb"] > best:
            best, best_t = s["mean_nu_lb"], float(t)
    a = (external[score_col].to_numpy(float) >= best_t).astype(float)
    return summarize_group_metrics(grouped_metrics(external, a, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap))["mean_nu_lb"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "data_splits.json"))
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "evaluation_protocol.json"))
    parser.add_argument("--bundle", default=str(ROOT / "artifacts" / "bundles" / "predictions_v1.csv"))
    parser.add_argument("--outdir", default=str(ROOT / "artifacts" / "results"))
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    bootstrap = protocol["primary_metric"]["bootstrap"]
    manifest = json.loads(Path(args.config).read_text(encoding="utf-8"))
    scen = {s["name"]: s for s in manifest["synthetic_scenarios"]}
    rows = []

    frame = generate_synthetic_scenario(scen["decision_relevant_state"])
    val = frame[frame["split"] == "validation"].reset_index(drop=True)
    ext = frame[frame["split"] == "external"].reset_index(drop=True).copy()
    calibrators = {
        "temperature": lambda p: apply_temperature(p, fit_temperature(val["p"].to_numpy(float), val["y"].to_numpy(float))),
        "platt": (lambda m: (lambda p: apply_platt(p, m)))(fit_platt(val["p"].to_numpy(float), val["y"].to_numpy(float))),
        "isotonic": (lambda m: (lambda p: apply_isotonic(p, m)))(fit_isotonic(val["p"].to_numpy(float), val["y"].to_numpy(float))),
    }
    val_raw = val.copy()
    ext_raw = ext.copy()
    raw_nu = _fixed_nu(val_raw, ext_raw, "p", RAW_GRID, bootstrap)
    for name, fn in calibrators.items():
        v = val.copy(); v["p"] = fn(val["p"].to_numpy(float))
        e = ext.copy(); e["p"] = fn(ext["p"].to_numpy(float))
        mapped_grid = fn(RAW_GRID)
        nu = _fixed_nu(v, e, "p", np.sort(mapped_grid), bootstrap)
        rows.append({"dataset": "synthetic_decision_relevant_state", "calibrator": name,
                     "raw_nu_lb": raw_nu, "calibrated_nu_lb": nu, "effect": nu - raw_nu})

    bundle = pd.read_csv(args.bundle, parse_dates=["date"], low_memory=False)
    real = bundle[bundle["asset"].isin(["BTC-USD", "TSLA"])].copy()
    btc = real[real["asset"] == "BTC-USD"]
    val = btc[btc["split"] == "validation"].reset_index(drop=True)
    ext = real[real["split"] == "external"].reset_index(drop=True).copy()
    calibrators = {
        "temperature": lambda p: apply_temperature(p, fit_temperature(val["p"].to_numpy(float), val["y"].to_numpy(float))),
        "platt": (lambda m: (lambda p: apply_platt(p, m)))(fit_platt(val["p"].to_numpy(float), val["y"].to_numpy(float))),
        "isotonic": (lambda m: (lambda p: apply_isotonic(p, m)))(fit_isotonic(val["p"].to_numpy(float), val["y"].to_numpy(float))),
    }
    val_raw = val.copy(); ext_raw = ext.copy()
    raw_nu = _fixed_nu(val_raw, ext_raw, "p", RAW_GRID, bootstrap)
    for name, fn in calibrators.items():
        v = val.copy(); v["p"] = fn(val["p"].to_numpy(float))
        e = ext.copy(); e["p"] = fn(ext["p"].to_numpy(float))
        nu = _fixed_nu(v, e, "p", np.sort(fn(RAW_GRID)), bootstrap)
        rows.append({"dataset": "real_BTC_TSLA", "calibrator": name,
                     "raw_nu_lb": raw_nu, "calibrated_nu_lb": nu, "effect": nu - raw_nu})

    result = pd.DataFrame(rows)
    out = Path(args.outdir) / "calibration_method_sensitivity.csv"
    result.to_csv(out, index=False)
    pd.set_option("display.width", 160)
    print(result.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))


if __name__ == "__main__":
    main()
