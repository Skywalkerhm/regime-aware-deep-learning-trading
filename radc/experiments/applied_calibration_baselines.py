#!/usr/bin/env python3
"""Group-level calibration baselines on the canonical real-asset bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rutl.evaluator import _current_rule_actions, _frame_for_split
from src.rutl.group_metrics import grouped_metrics, summarize_group_metrics


def logits(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(p / (1.0 - p))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    z = logits(probabilities)
    y = np.asarray(labels, dtype=float)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        p = sigmoid(z / temperature)
        return -float(np.mean(y * np.log(p + 1e-12) + (1.0 - y) * np.log(1.0 - p + 1e-12)))

    result = minimize_scalar(objective, bounds=(-2.0, 2.0), method="bounded")
    return float(np.exp(result.x))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    return sigmoid(logits(probabilities) / temperature)


def fit_platt(probabilities: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits(probabilities).reshape(-1, 1), labels)
    return model


def apply_platt(probabilities: np.ndarray, model: LogisticRegression) -> np.ndarray:
    return model.predict_proba(logits(probabilities).reshape(-1, 1))[:, 1]


def fit_isotonic(probabilities: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip", increasing=True)
    model.fit(probabilities, labels)
    return model


def apply_isotonic(probabilities: np.ndarray, model: IsotonicRegression) -> np.ndarray:
    return model.transform(probabilities)


def ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(probabilities, edges) - 1, 0, bins - 1)
    value = 0.0
    for i in range(bins):
        mask = indices == i
        if mask.any():
            value += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return value


def choose_group_threshold(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
    protocol: dict,
    transaction_cost: float,
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_summary: dict[str, float] | None = None
    bootstrap = protocol["primary_metric"]["bootstrap"]
    for threshold in np.linspace(0.30, 0.80, 51):
        actions = (probabilities >= float(threshold)).astype(float)
        groups = grouped_metrics(
            validation,
            actions,
            transaction_cost=transaction_cost,
            drawdown_penalty=float(protocol["utility"]["drawdown_penalty"]),
            bootstrap=bootstrap,
        )
        summary = summarize_group_metrics(groups)
        if best_summary is None or summary["mean_nu_lb"] > best_summary["mean_nu_lb"]:
            best_threshold = float(threshold)
            best_summary = summary
    if best_summary is None:
        raise RuntimeError("no threshold selected")
    return best_threshold, best_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(ROOT / "artifacts" / "bundles" / "predictions_v1.csv"))
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "evaluation_protocol.json"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "results" / "applied_calibration_baselines_v1.csv"))
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    bundle = pd.read_csv(args.bundle, parse_dates=["date"], low_memory=False)
    validation = _frame_for_split(bundle, "validation")
    validation = validation[validation["asset"] == "BTC-USD"].copy().sort_values(["model", "date"]).reset_index(drop=True)
    external = _frame_for_split(bundle, "external")
    external = external[external["asset"].isin(["BTC-USD", "TSLA"])].copy().sort_values(["model", "date"]).reset_index(drop=True)
    transaction_cost = float(protocol["costs"]["transaction_cost_bps"]) / 10000.0
    bootstrap = protocol["primary_metric"]["bootstrap"]

    validation_calibrated: dict[tuple[str, str], np.ndarray] = {}
    external_calibrated: dict[tuple[str, str], np.ndarray] = {}
    methods = ["raw", "temperature", "platt", "isotonic"]
    for model in sorted(validation["model"].unique()):
        val = validation[validation["model"] == model].reset_index(drop=True)
        labels = val["y"].to_numpy(dtype=float)
        raw_val = val["p"].to_numpy(dtype=float)
        temperature = fit_temperature(raw_val, labels)
        platt = fit_platt(raw_val, labels)
        isotonic = fit_isotonic(raw_val, labels)
        validation_calibrated[(model, "raw")] = raw_val
        validation_calibrated[(model, "temperature")] = apply_temperature(raw_val, temperature)
        validation_calibrated[(model, "platt")] = apply_platt(raw_val, platt)
        validation_calibrated[(model, "isotonic")] = apply_isotonic(raw_val, isotonic)
        test = external[external["model"] == model].reset_index(drop=True)
        raw_test = test["p"].to_numpy(dtype=float)
        external_calibrated[(model, "raw")] = raw_test
        external_calibrated[(model, "temperature")] = apply_temperature(raw_test, temperature)
        external_calibrated[(model, "platt")] = apply_platt(raw_test, platt)
        external_calibrated[(model, "isotonic")] = apply_isotonic(raw_test, isotonic)

    rows: list[dict[str, object]] = []
    for method in methods:
        val = validation.copy()
        val["calibrated_p"] = np.concatenate([
            validation_calibrated[(model, method)]
            for model in sorted(validation["model"].unique())
        ])
        test = external.copy()
        test["calibrated_p"] = np.concatenate([
            external_calibrated[(model, method)]
            for model in sorted(validation["model"].unique())
        ])
        selected_threshold, _ = choose_group_threshold(
            val,
            val["calibrated_p"].to_numpy(),
            protocol,
            transaction_cost,
        )
        policies = {
            "fixed_0.50": (test["calibrated_p"].to_numpy() >= 0.50).astype(float),
            "validation_optimized": (test["calibrated_p"].to_numpy() >= selected_threshold).astype(float),
            "regime_mapping": _current_rule_actions(test.assign(p=test["calibrated_p"])),
        }
        for policy_name, actions in policies.items():
            groups = grouped_metrics(
                test,
                actions,
                transaction_cost=transaction_cost,
                drawdown_penalty=float(protocol["utility"]["drawdown_penalty"]),
                bootstrap=bootstrap,
            )
            summary = summarize_group_metrics(groups)
            rows.append({
                "calibration": method,
                "threshold_policy": policy_name,
                "threshold": 0.50 if policy_name == "fixed_0.50" else selected_threshold if policy_name == "validation_optimized" else np.nan,
                "ece_external": ece(test["calibrated_p"].to_numpy(), test["y"].to_numpy()),
                **summary,
            })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
