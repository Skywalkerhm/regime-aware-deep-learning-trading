#!/usr/bin/env python3
"""Group-level robustness analyses for the applied regime-threshold paper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rutl.evaluator import _current_rule_actions, _fixed_threshold_actions, _frame_for_split
from src.rutl.group_metrics import grouped_metrics, select_fixed_threshold_grouped, summarize_group_metrics

REGIME_COLUMNS = [
    "posterior_crisis",
    "posterior_trend_up",
    "posterior_trend_down",
    "posterior_chop",
]


def paired_group_bootstrap(differences: np.ndarray, *, replications: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(differences, dtype=float)
    samples = rng.choice(values, size=(replications, len(values)), replace=True).mean(axis=1)
    p_value = float((np.sum(samples <= 0.0) + 1.0) / (replications + 1.0))
    return float(np.quantile(samples, 0.05)), float(np.quantile(samples, 0.95)), p_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(ROOT / "artifacts" / "bundles" / "predictions_v1.csv"))
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "evaluation_protocol.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "results"))
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    bundle = pd.read_csv(args.bundle, parse_dates=["date"], low_memory=False)
    validation = _frame_for_split(bundle, "validation")
    validation = validation[validation["asset"] == "BTC-USD"].copy()
    external_all = _frame_for_split(bundle, "external")
    external = external_all[external_all["asset"].isin(["BTC-USD", "TSLA"])].copy()
    bootstrap = protocol["primary_metric"]["bootstrap"]
    tc = float(protocol["costs"]["transaction_cost_bps"]) / 10000.0
    dd = float(protocol["utility"]["drawdown_penalty"])
    fixed_threshold, _, _ = select_fixed_threshold_grouped(
        validation,
        grid=np.linspace(0.30, 0.80, 51),
        transaction_cost=tc,
        drawdown_penalty=dd,
        bootstrap=bootstrap,
        score_column="p",
    )

    # External group table and equal-weight group bootstrap.
    fixed_groups = grouped_metrics(external, (external["p"].to_numpy(dtype=float) >= fixed_threshold).astype(float), transaction_cost=tc, drawdown_penalty=dd, bootstrap=bootstrap)
    rule_groups = grouped_metrics(external, _current_rule_actions(external), transaction_cost=tc, drawdown_penalty=dd, bootstrap=bootstrap)
    merged = fixed_groups[["asset", "fold", "model", "nu_lb", "sharpe", "max_drawdown", "turnover"]].merge(
        rule_groups[["asset", "fold", "model", "nu_lb", "sharpe", "max_drawdown", "turnover"]],
        on=["asset", "fold", "model"],
        suffixes=("_fixed", "_rule"),
    )
    merged["difference"] = merged["nu_lb_rule"] - merged["nu_lb_fixed"]
    low, high, p_value = paired_group_bootstrap(
        merged["difference"].to_numpy(),
        replications=int(bootstrap["replications"]),
        seed=int(bootstrap["seed"]),
    )
    group_summary = pd.DataFrame([{
        "n_groups": len(merged),
        "mean_difference": float(merged["difference"].mean()),
        "median_difference": float(merged["difference"].median()),
        "wins": int((merged["difference"] > 0).sum()),
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
    }])
    merged.to_csv(Path(args.output_dir) / "applied_external_groups_v1.csv", index=False)
    group_summary.to_csv(Path(args.output_dir) / "applied_group_bootstrap_v1.csv", index=False)

    # Static baselines, summarized across groups.
    baseline_rows: list[dict[str, object]] = []
    for name, actions in (
        ("fixed_0.50", (external["p"].to_numpy(dtype=float) >= 0.50).astype(float)),
        ("fixed_validation_selected", (external["p"].to_numpy(dtype=float) >= fixed_threshold).astype(float)),
        ("current_rule", _current_rule_actions(external)),
    ):
        groups = grouped_metrics(external, actions, transaction_cost=tc, drawdown_penalty=dd, bootstrap=bootstrap)
        baseline_rows.append({"policy": name, **summarize_group_metrics(groups)})
    pd.DataFrame(baseline_rows).to_csv(Path(args.output_dir) / "applied_static_baselines_v1.csv", index=False)

    # Cost sensitivity, summarized across groups.
    cost_rows: list[dict[str, object]] = []
    for cost_bps in [0, 5, 10, 20, 50, 100]:
        cost = cost_bps / 10000.0
        for name, actions in (
            ("fixed_validation_selected", (external["p"].to_numpy(dtype=float) >= fixed_threshold).astype(float)),
            ("current_rule", _current_rule_actions(external)),
        ):
            groups = grouped_metrics(external, actions, transaction_cost=cost, drawdown_penalty=dd, bootstrap=bootstrap)
            cost_rows.append({"cost_bps": cost_bps, "policy": name, **summarize_group_metrics(groups)})
    pd.DataFrame(cost_rows).to_csv(Path(args.output_dir) / "applied_cost_sensitivity_v1.csv", index=False)

    # Placebo: permute regime posteriors within each asset/model/fold group.
    rng = np.random.default_rng(int(bootstrap["seed"]))
    placebo_rows: list[dict[str, object]] = []
    for keys, indices in external.groupby(["asset", "fold", "model"]).groups.items():
        frame = external.loc[indices].sort_values("date").reset_index(drop=True)
        fixed_group = grouped_metrics(frame, (frame["p"].to_numpy(dtype=float) >= fixed_threshold).astype(float), transaction_cost=tc, drawdown_penalty=dd, bootstrap=bootstrap)
        real_group = grouped_metrics(frame, _current_rule_actions(frame), transaction_cost=tc, drawdown_penalty=dd, bootstrap=bootstrap)
        fixed_value = float(fixed_group["nu_lb"].iloc[0])
        real_delta = float(real_group["nu_lb"].iloc[0]) - fixed_value
        shuffled = []
        for _ in range(200):
            permuted = frame.copy()
            order = rng.permutation(len(permuted))
            permuted[REGIME_COLUMNS] = permuted[REGIME_COLUMNS].to_numpy()[order]
            shuffled_groups = grouped_metrics(permuted, _current_rule_actions(permuted), transaction_cost=tc, drawdown_penalty=dd, bootstrap=bootstrap)
            shuffled.append(float(shuffled_groups["nu_lb"].iloc[0]) - fixed_value)
        shuffled = np.asarray(shuffled)
        asset, fold, model = keys
        placebo_rows.append({
            "asset": asset,
            "fold": fold,
            "model": model,
            "real_delta": real_delta,
            "shuffle_mean": float(shuffled.mean()),
            "shuffle_sd": float(shuffled.std(ddof=1)),
            "percentile": float((np.sum(shuffled < real_delta) + 1.0) / (len(shuffled) + 1.0)),
            "n_shuffle": len(shuffled),
        })
    pd.DataFrame(placebo_rows).to_csv(Path(args.output_dir) / "applied_placebo_rule_labels_v1.csv", index=False)

    print(json.dumps({
        "fixed_threshold": fixed_threshold,
        "group_bootstrap": group_summary.iloc[0].to_dict(),
        "static_baselines": baseline_rows,
        "cost_sensitivity": cost_rows,
        "placebo_groups": len(placebo_rows),
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
