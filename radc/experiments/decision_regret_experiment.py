#!/usr/bin/env python3
"""Experiment A: decision-regret curves and the decision-calibration mechanism.

Confirms whether "Regime-Aware Decision Calibration" is the correct mechanism by
checking, on synthetic DGPs and the existing BTC/TSLA bundle:

  1. Decision regret reduction is positive only when state information changes the
     optimal action with positive economic value.
  2. Decision regret reduction -- not probability calibration (ECE / Brier) --
     tracks the realized out-of-sample utility gain of a regime threshold.
  3. Uninformative states kill both the regret reduction and the realized gain.

No new assets are used. This re-uses the frozen BTC/TSLA prediction bundle and the
existing synthetic DGPs only.
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
    _bin_index,
    brier_score,
    build_delta_table,
    default_score_edges,
    decision_regret,
    detector_states,
    expected_calibration_error,
    regret_of_threshold,
    regret_of_threshold_map,
    select_fixed_threshold_by_regret,
    select_state_thresholds_by_regret,
    true_states,
)
from src.rutl.group_metrics import grouped_metrics, select_fixed_threshold_grouped, summarize_group_metrics

SCORE_GRID = np.linspace(0.05, 0.95, 46)
NU_GRID = np.linspace(0.30, 0.80, 51)


def _delta_state_contrast(table) -> float:
    """Between-state dispersion of Delta(p, z) inside each score bin.

    A policy-free measure of decision relevance: ~0 when the optimal action is
    state-invariant, largest when states demand different actions at the same
    forecast probability.
    """
    bins = sorted({b for (b, _) in table.cell})
    contrasts = []
    for b in bins:
        values = [table.cell[(bb, s)] for (bb, s) in table.cell if bb == b]
        if len(values) >= 2:
            contrasts.append(max(values) - min(values))
    return float(np.mean(contrasts)) if contrasts else 0.0


def _best_state_thresholds(part: pd.DataFrame, state_column: str) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for state in sorted(part[state_column].unique()):
        block = part[part[state_column] == state]
        best_t, best_score = float(NU_GRID[0]), -1e9
        for threshold in NU_GRID:
            actions = (block["p"].to_numpy(dtype=float) >= float(threshold)).astype(float)
            result = evaluate_actions(actions, block["r_next"].to_numpy(dtype=float), transaction_cost=0.001, drawdown_penalty=0.25)
            score = result.annualized_mean - 0.05 * result.annualized_vol - 0.10 * result.max_drawdown
            if score > best_score:
                best_score, best_t = score, float(threshold)
        thresholds[int(state)] = best_t
    return thresholds


def _realized_oracle_gain(
    search: pd.DataFrame,
    validation: pd.DataFrame,
    external: pd.DataFrame,
    *,
    protocol: dict,
    state_column: str,
) -> dict[str, float]:
    """Realized external utility of a state threshold vs the validation-selected fixed threshold.

    The fixed threshold is selected on validation by group-level NU_LB (applied
    protocol); per-state thresholds are selected on the search split by the same
    utility proxy. Both are then applied to the external split, so no external data
    is used for selection.
    """
    bootstrap = protocol["primary_metric"]["bootstrap"]
    fixed_threshold, _, _ = select_fixed_threshold_grouped(
        validation,
        grid=NU_GRID,
        transaction_cost=0.001,
        drawdown_penalty=0.25,
        bootstrap=bootstrap,
        score_column="p",
    )
    thresholds = _best_state_thresholds(search, state_column)
    states = external[state_column].to_numpy(dtype=int)
    default = float(np.mean(list(thresholds.values())))
    cutoffs = np.array([thresholds.get(int(s), default) for s in states], dtype=float)
    p_ext = external["p"].to_numpy(dtype=float)
    state_actions = (p_ext >= cutoffs).astype(float)
    fixed_actions = (p_ext >= float(fixed_threshold)).astype(float)
    state_summary = summarize_group_metrics(
        grouped_metrics(external, state_actions, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
    )
    fixed_summary = summarize_group_metrics(
        grouped_metrics(external, fixed_actions, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
    )
    return {
        "fixed_threshold": float(fixed_threshold),
        "fixed_nu_lb": fixed_summary["mean_nu_lb"],
        "state_nu_lb": state_summary["mean_nu_lb"],
        "realized_state_gain": state_summary["mean_nu_lb"] - fixed_summary["mean_nu_lb"],
    }


def _analyze(
    train: pd.DataFrame,
    search: pd.DataFrame,
    validation: pd.DataFrame,
    external: pd.DataFrame,
    *,
    protocol: dict,
    state_column: str,
    edges: np.ndarray,
) -> dict[str, float]:
    if train.empty or external.empty:
        raise ValueError("train and external frames must be non-empty")

    states_train = train[state_column].to_numpy(dtype=int)
    states_ext = external[state_column].to_numpy(dtype=int)

    train_table = build_delta_table(train, states_train, edges=edges)
    oracle_table = build_delta_table(external, states_ext, edges=edges)

    # Deployable: thresholds fit on train regret, evaluated on external regret.
    theta_fixed_deploy = select_fixed_threshold_by_regret(train, train_table, edges=edges, grid=SCORE_GRID, states=states_train)
    theta_state_deploy = select_state_thresholds_by_regret(train, train_table, edges=edges, grid=SCORE_GRID, states=states_train)
    regret_fixed_deploy = regret_of_threshold(external, theta_fixed_deploy, train_table, edges=edges, state_override=states_ext)
    regret_state_deploy = regret_of_threshold_map(external, theta_state_deploy, train_table, edges=edges, state_override=states_ext)

    # Oracle: fit and evaluate regret in-sample on external (reference ceiling).
    theta_fixed_oracle = select_fixed_threshold_by_regret(external, oracle_table, edges=edges, grid=SCORE_GRID, states=states_ext)
    theta_state_oracle = select_state_thresholds_by_regret(external, oracle_table, edges=edges, grid=SCORE_GRID, states=states_ext)
    regret_fixed_oracle = regret_of_threshold(external, theta_fixed_oracle, oracle_table, edges=edges, state_override=states_ext)
    regret_state_oracle = regret_of_threshold_map(external, theta_state_oracle, oracle_table, edges=edges, state_override=states_ext)

    p_ext = external["p"].to_numpy(dtype=float)
    y_ext = external["y"].to_numpy(dtype=float)
    realized = _realized_oracle_gain(search, validation, external, protocol=protocol, state_column=state_column)

    return {
        "n_external": float(len(external)),
        "theta_fixed_deploy": float(theta_fixed_deploy),
        "regret_fixed_deploy": regret_fixed_deploy,
        "regret_state_deploy": regret_state_deploy,
        "regret_reduction_deploy": regret_fixed_deploy - regret_state_deploy,
        "regret_fixed_oracle": regret_fixed_oracle,
        "regret_state_oracle": regret_state_oracle,
        "regret_reduction_oracle": regret_fixed_oracle - regret_state_oracle,
        "delta_state_contrast": _delta_state_contrast(oracle_table),
        "ece": expected_calibration_error(p_ext, y_ext),
        "brier": brier_score(p_ext, y_ext),
        **{f"realized_{k}": v for k, v in realized.items()},
    }


def _split(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame[frame["split"] == name].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "data_splits.json"))
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "evaluation_protocol.json"))
    parser.add_argument("--bundle", default=str(ROOT / "artifacts" / "bundles" / "predictions_v1.csv"))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--outdir", default=str(ROOT / "artifacts" / "results"))
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.config).read_text(encoding="utf-8"))
    scenarios = {s["name"]: s for s in manifest["synthetic_scenarios"]}
    edges = default_score_edges(10)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for name, base in scenarios.items():
        for offset in range(args.repeats):
            scenario = dict(base)
            scenario["seed"] = int(base["seed"]) + offset * 101
            frame = generate_synthetic_scenario(scenario)
            frame["state_true"] = true_states(frame)
            frame["state_det"] = detector_states(frame)
            search = _split(frame, "search")
            validation = _split(frame, "validation")
            external = _split(frame, "external")
            train = pd.concat([search, validation], ignore_index=True)
            result = _analyze(train, search, validation, external, protocol=protocol, state_column="state_true", edges=edges)
            rows.append({"dataset": "synthetic_true_state", "scenario": name, "seed": scenario["seed"], **result})

    synthetic = pd.DataFrame(rows)
    synthetic.to_csv(outdir / "decision_regret_synthetic.csv", index=False)

    # Real data: train on BTC development folds; detector state is a labelled
    # pseudo-oracle proxy. External subsets are evaluated independently.
    bundle = pd.read_csv(args.bundle, parse_dates=["date"], low_memory=False)
    real = bundle[bundle["asset"].isin(["BTC-USD", "TSLA"])].copy()
    real["state_det"] = detector_states(real)
    btc = real[real["asset"] == "BTC-USD"]
    btc_search = _split(btc, "search")
    btc_validation = _split(btc, "validation")
    btc_train = pd.concat([btc_search, btc_validation], ignore_index=True)
    external_all = _split(real, "external")
    real_rows = []
    subsets = [("ALL", external_all)]
    subsets += [(a, external_all[external_all["asset"] == a].reset_index(drop=True)) for a in sorted(external_all["asset"].unique())]
    for label, part in subsets:
        result = _analyze(btc_train, btc_search, btc_validation, part, protocol=protocol, state_column="state_det", edges=edges)
        real_rows.append({"dataset": "real_detector_proxy", "scenario": label, "seed": 20260928, **result})
    real_df = pd.DataFrame(real_rows)
    real_df.to_csv(outdir / "decision_regret_real.csv", index=False)

    # Decision-regret curves for a positive and a negative control (mean over seeds).
    curve_rows = []
    for name in ["decision_relevant_state", "uninformative_regime"]:
        base = scenarios[name]
        acc = []
        for offset in range(args.repeats):
            scenario = dict(base)
            scenario["seed"] = int(base["seed"]) + offset * 101
            frame = generate_synthetic_scenario(scenario)
            frame["state_true"] = true_states(frame)
            external = _split(frame, "external")
            table = build_delta_table(external, external["state_true"].to_numpy(dtype=int), edges=edges)
            bins = _bin_index(external["p"].to_numpy(dtype=float), edges)
            states = external["state_true"].to_numpy(dtype=int)
            for threshold in SCORE_GRID:
                actions = (external["p"].to_numpy(dtype=float) >= float(threshold)).astype(float)
                acc.append({"scenario": name, "threshold": float(threshold), "regret": decision_regret(bins, states, actions, table)})
        curve = pd.DataFrame(acc).groupby(["scenario", "threshold"], as_index=False)["regret"].mean()
        curve_rows.append(curve)
    pd.concat(curve_rows, ignore_index=True).to_csv(outdir / "decision_regret_curves.csv", index=False)

    def _corr(a: pd.Series, b: pd.Series) -> float:
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a.to_numpy(dtype=float), b.to_numpy(dtype=float))[0, 1])

    summary = {
        "n_rows": int(len(synthetic)),
        "corr_regret_reduction_oracle__realized_gain": _corr(synthetic["regret_reduction_oracle"], synthetic["realized_realized_state_gain"]),
        "corr_delta_state_contrast__realized_gain": _corr(synthetic["delta_state_contrast"], synthetic["realized_realized_state_gain"]),
        "corr_ece__realized_gain": _corr(synthetic["ece"], synthetic["realized_realized_state_gain"]),
        "corr_brier__realized_gain": _corr(synthetic["brier"], synthetic["realized_realized_state_gain"]),
        "corr_ece__regret_reduction_oracle": _corr(synthetic["ece"], synthetic["regret_reduction_oracle"]),
        "by_scenario": synthetic.groupby("scenario").agg(
            regret_reduction_oracle=("regret_reduction_oracle", "mean"),
            regret_reduction_deploy=("regret_reduction_deploy", "mean"),
            delta_state_contrast=("delta_state_contrast", "mean"),
            ece=("ece", "mean"),
            brier=("brier", "mean"),
            realized_gain=("realized_realized_state_gain", "mean"),
        ).reset_index().to_dict(orient="records"),
    }
    (outdir / "decision_regret_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["by_scenario"], indent=2))
    print("\nCorrelations:")
    for key in [k for k in summary if k.startswith("corr_")]:
        print(f"  {key} = {summary[key]:+.3f}")
    print("\nReal (detector proxy):")
    print(real_df[["scenario", "regret_reduction_deploy", "regret_reduction_oracle", "delta_state_contrast", "realized_realized_state_gain"]].to_string(index=False))


if __name__ == "__main__":
    main()
