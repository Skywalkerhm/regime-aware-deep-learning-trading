#!/usr/bin/env python3
"""Validation-tuned regime thresholds on a fair group-level protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rutl.evaluator import _frame_for_split
from src.rutl.group_metrics import grouped_metrics, select_fixed_threshold_grouped, summarize_group_metrics

STATES = ["crisis", "trend_up", "trend_down", "chop"]


def tuned_rule_actions(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    regime = np.argmax(
        frame[["posterior_crisis", "posterior_trend_up", "posterior_trend_down", "posterior_chop"]].to_numpy(),
        axis=1,
    )
    cutoff = np.array([thresholds[STATES[i]] for i in regime], dtype=float)
    return (frame["p"].to_numpy(dtype=float) >= cutoff).astype(float)


def score_thresholds(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    *,
    protocol: dict,
    transaction_cost: float,
) -> tuple[float, pd.DataFrame]:
    groups = grouped_metrics(
        frame,
        tuned_rule_actions(frame, thresholds),
        transaction_cost=transaction_cost,
        drawdown_penalty=float(protocol["utility"]["drawdown_penalty"]),
        bootstrap=protocol["primary_metric"]["bootstrap"],
    )
    return summarize_group_metrics(groups)["mean_nu_lb"], groups


def coordinate_tune(
    validation: pd.DataFrame,
    *,
    initial: dict[str, float],
    protocol: dict,
    transaction_cost: float,
    passes: int = 4,
) -> tuple[dict[str, float], float]:
    thresholds = dict(initial)
    best_score, _ = score_thresholds(validation, thresholds, protocol=protocol, transaction_cost=transaction_cost)
    grid = np.round(np.arange(0.20, 0.86, 0.02), 2)
    for _ in range(passes):
        changed = False
        for state in STATES:
            local_best = thresholds[state]
            local_score = best_score
            for value in grid:
                candidate = dict(thresholds)
                candidate[state] = float(value)
                score, _ = score_thresholds(validation, candidate, protocol=protocol, transaction_cost=transaction_cost)
                if score > local_score + 1e-12:
                    local_score = score
                    local_best = float(value)
            if local_best != thresholds[state]:
                thresholds[state] = local_best
                best_score = local_score
                changed = True
        if not changed:
            break
    return thresholds, best_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(ROOT / "artifacts" / "bundles" / "predictions_v1.csv"))
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "evaluation_protocol.json"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "results" / "applied_tuned_regime_v1.csv"))
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    bundle = pd.read_csv(args.bundle, parse_dates=["date"], low_memory=False)
    validation = _frame_for_split(bundle, "validation")
    validation = validation[validation["asset"] == "BTC-USD"].copy()
    external = _frame_for_split(bundle, "external")
    external = external[external["asset"].isin(["BTC-USD", "TSLA"])].copy()
    tc = float(protocol["costs"]["transaction_cost_bps"]) / 10000.0
    bootstrap = protocol["primary_metric"]["bootstrap"]

    fixed_threshold, _, _ = select_fixed_threshold_grouped(
        validation,
        grid=np.linspace(0.30, 0.80, 51),
        transaction_cost=tc,
        drawdown_penalty=float(protocol["utility"]["drawdown_penalty"]),
        bootstrap=bootstrap,
        score_column="p",
    )

    initial = {"crisis": 0.80, "trend_up": 0.40, "trend_down": 0.50, "chop": 0.55}
    global_thresholds, global_validation_score = coordinate_tune(
        validation,
        initial=initial,
        protocol=protocol,
        transaction_cost=tc,
    )
    global_groups = grouped_metrics(
        external,
        tuned_rule_actions(external, global_thresholds),
        transaction_cost=tc,
        drawdown_penalty=float(protocol["utility"]["drawdown_penalty"]),
        bootstrap=bootstrap,
    )
    global_summary = summarize_group_metrics(global_groups)

    fixed_groups = grouped_metrics(
        external,
        (external["p"].to_numpy(dtype=float) >= fixed_threshold).astype(float),
        transaction_cost=tc,
        drawdown_penalty=float(protocol["utility"]["drawdown_penalty"]),
        bootstrap=bootstrap,
    )
    fixed_summary = summarize_group_metrics(fixed_groups)

    model_rows: list[dict[str, object]] = []
    model_external_parts: list[pd.DataFrame] = []
    for model in sorted(validation["model"].unique()):
        val = validation[validation["model"] == model].copy()
        thresholds, val_score = coordinate_tune(
            val,
            initial=initial,
            protocol=protocol,
            transaction_cost=tc,
        )
        test = external[external["model"] == model].copy()
        groups = grouped_metrics(
            test,
            tuned_rule_actions(test, thresholds),
            transaction_cost=tc,
            drawdown_penalty=float(protocol["utility"]["drawdown_penalty"]),
            bootstrap=bootstrap,
        )
        model_external_parts.append(groups)
        model_rows.append({
            "model": model,
            "validation_score": val_score,
            **thresholds,
            **{f"external_{k}": v for k, v in summarize_group_metrics(groups).items()},
        })
    per_model_external = pd.concat(model_external_parts, ignore_index=True)
    per_model_summary = summarize_group_metrics(per_model_external)

    rows = [
        {"policy": "current_prior", **summarize_group_metrics(grouped_metrics(external, tuned_rule_actions(external, initial), transaction_cost=tc, drawdown_penalty=float(protocol["utility"]["drawdown_penalty"]), bootstrap=bootstrap))},
        {"policy": "validation_tuned_global", **global_summary},
        {"policy": "validation_tuned_per_model", **per_model_summary},
        {"policy": "fixed_validation_selected", **fixed_summary},
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    pd.DataFrame(model_rows).to_csv(output.with_name("applied_tuned_regime_per_model_v1.csv"), index=False)
    print(json.dumps({
        "fixed_threshold": fixed_threshold,
        "global_thresholds": global_thresholds,
        "global_validation_score": global_validation_score,
        "summary": rows,
        "per_model": model_rows,
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
