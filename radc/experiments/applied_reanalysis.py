#!/usr/bin/env python3
"""Canonical group-level applied reanalysis: fixed threshold versus rule threshold."""

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
from src.rutl.group_metrics import (
    grouped_metrics,
    select_fixed_threshold_grouped,
    summarize_group_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(ROOT / "artifacts" / "bundles" / "predictions_v1.csv"))
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "evaluation_protocol.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "results"))
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    bundle = pd.read_csv(args.bundle, parse_dates=["date"], low_memory=False)
    search = _frame_for_split(bundle, "search")
    validation = _frame_for_split(bundle, "validation")
    external_all = _frame_for_split(bundle, "external")
    search = search[search["asset"] == "BTC-USD"].copy()
    validation = validation[validation["asset"] == "BTC-USD"].copy()
    external = external_all[external_all["asset"].isin(["BTC-USD", "TSLA"])].copy()

    transaction_cost = float(protocol["costs"]["transaction_cost_bps"]) / 10000.0
    drawdown_penalty = float(protocol["utility"]["drawdown_penalty"])
    bootstrap = protocol["primary_metric"]["bootstrap"]
    fixed_threshold, _, _ = select_fixed_threshold_grouped(
        validation,
        grid=np.linspace(0.30, 0.80, 51),
        transaction_cost=transaction_cost,
        drawdown_penalty=drawdown_penalty,
        bootstrap=bootstrap,
        score_column="p",
    )

    def policy_groups(frame: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        fixed = grouped_metrics(
            frame,
            (frame["p"].to_numpy(dtype=float) >= threshold).astype(float),
            transaction_cost=transaction_cost,
            drawdown_penalty=drawdown_penalty,
            bootstrap=bootstrap,
        )
        rule = grouped_metrics(
            frame,
            _current_rule_actions(frame),
            transaction_cost=transaction_cost,
            drawdown_penalty=drawdown_penalty,
            bootstrap=bootstrap,
        )
        return fixed, rule

    datasets = {
        "search": search,
        "validation": validation,
        "external": external,
    }
    summary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    all_group_rows: list[pd.DataFrame] = []
    external_groups: pd.DataFrame | None = None
    rng = np.random.default_rng(int(bootstrap["seed"]))

    for scope, frame in datasets.items():
        fixed, rule = policy_groups(frame, fixed_threshold)
        fixed["scope"] = scope
        fixed["policy"] = "fixed_validation_optimized"
        rule["scope"] = scope
        rule["policy"] = "current_rule"
        all_group_rows.extend([fixed, rule])
        for name, groups in (("fixed_validation_optimized", fixed), ("current_rule", rule)):
            summary_rows.append({"scope": scope, "policy": name, "fixed_threshold": fixed_threshold, **summarize_group_metrics(groups)})
        merged = fixed[["asset", "fold", "model", "nu_lb"]].merge(
            rule[["asset", "fold", "model", "nu_lb"]],
            on=["asset", "fold", "model"],
            suffixes=("_fixed", "_rule"),
        )
        merged["difference"] = merged["nu_lb_rule"] - merged["nu_lb_fixed"]
        differences = merged["difference"].to_numpy(dtype=float)
        samples = rng.choice(
            differences,
            size=(int(bootstrap["replications"]), len(differences)),
            replace=True,
        ).mean(axis=1)
        paired_rows.append({
            "scope": scope,
            "fixed_threshold": fixed_threshold,
            "n_groups": len(merged),
            "mean_difference": float(differences.mean()),
            "median_difference": float(np.median(differences)),
            "wins": int((differences > 0).sum()),
            "ci_low": float(np.quantile(samples, 0.05)),
            "ci_high": float(np.quantile(samples, 0.95)),
            "p": float((np.sum(samples <= 0.0) + 1.0) / (len(samples) + 1.0)),
        })
        if scope == "external":
            external_groups = merged

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output / "applied_fixed_vs_rule_v1.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(output / "applied_paired_bootstrap_v1.csv", index=False)
    pd.concat(all_group_rows, ignore_index=True).to_csv(output / "applied_group_metrics_v1.csv", index=False)
    if external_groups is None:
        raise RuntimeError("external group comparison missing")
    external_groups.to_csv(output / "applied_external_by_fold_model_v1.csv", index=False)
    print(json.dumps({
        "fixed_threshold": fixed_threshold,
        "summaries": summary_rows,
        "paired": paired_rows,
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
