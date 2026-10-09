#!/usr/bin/env python3
"""Experiment D: state-shuffle placebo and bootstrap split stability under RADC.

Part 1 -- placebo. For each synthetic scenario, compute the realized gain of the
true-state threshold policy. Then destroy the state-outcome alignment with two
placebos that preserve the marginal state distribution and persistence:

  * circular_shift : rotate the state sequence by a random block-aligned offset;
  * block_permute  : permute consecutive time blocks of state labels.

A genuine decision-relevant state must lose its gain under both placebos. A
decision-irrelevant state (statistically separated but utility-invariant) already
has ~zero gain, so the placebo changes nothing for it.

Part 2 -- stability. Bootstrap-resample the search split and record how often the
DRRL split selection is stable, plus which context feature is chosen, for the
positive control versus the negative controls. No new assets are used.
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
from src.rutl.drrl import fit_stable_split_policy
from src.rutl.group_metrics import grouped_metrics, summarize_group_metrics

NU_GRID = np.linspace(0.30, 0.80, 51)
BLOCK = 50


def _best_state_thresholds(part: pd.DataFrame, state_col: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for state in sorted(part[state_col].unique()):
        block = part[part[state_col] == state]
        best_t, best = float(NU_GRID[0]), -1e9
        for threshold in NU_GRID:
            actions = (block["p"].to_numpy(float) >= float(threshold)).astype(float)
            result = evaluate_actions(actions, block["r_next"].to_numpy(float), transaction_cost=0.001, drawdown_penalty=0.25)
            score = result.annualized_mean - 0.05 * result.annualized_vol - 0.10 * result.max_drawdown
            if score > best:
                best, best_t = score, float(threshold)
        out[int(state)] = best_t
    return out


def _actions(frame: pd.DataFrame, thresholds: dict[int, float], states: np.ndarray) -> np.ndarray:
    default = float(np.mean(list(thresholds.values())))
    cutoffs = np.array([thresholds.get(int(s), default) for s in states], dtype=float)
    return (frame["p"].to_numpy(float) >= cutoffs).astype(float)


def _gain(frame: pd.DataFrame, thresholds: dict[int, float], states: np.ndarray, fixed_nu: float, bootstrap: dict) -> float:
    actions = _actions(frame, thresholds, states)
    nu = summarize_group_metrics(
        grouped_metrics(frame, actions, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
    )["mean_nu_lb"]
    return nu - fixed_nu


def _circular_shift(states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(states)
    offsets = np.arange(BLOCK, n - BLOCK + 1, BLOCK)
    if len(offsets) == 0:
        return rng.permutation(states)
    return np.roll(states, int(rng.choice(offsets)))


def _block_permute(states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(states)
    blocks = [states[i:i + BLOCK] for i in range(0, n, BLOCK)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "data_splits.json"))
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "evaluation_protocol.json"))
    parser.add_argument("--drrl-config", default=str(ROOT / "direction_evolve2" / "experiment_b_drrl_config.json"))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--stability-seeds", type=int, default=15)
    parser.add_argument("--outdir", default=str(ROOT / "artifacts" / "results"))
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    drrl_config = json.loads(Path(args.drrl_config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.config).read_text(encoding="utf-8"))
    scenarios = {s["name"]: s for s in manifest["synthetic_scenarios"]}
    bootstrap = protocol["primary_metric"]["bootstrap"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    placebo_rows: list[dict] = []
    stability_rows: list[dict] = []
    for name in ["decision_relevant_state", "asymmetric_utility", "high_separation", "uninformative_regime"]:
        base = scenarios[name]
        for offset in range(args.repeats):
            scenario = dict(base)
            scenario["seed"] = int(base["seed"]) + offset * 101
            frame = generate_synthetic_scenario(scenario)
            frame["state_true"] = true_states(frame)
            search = frame[frame["split"] == "search"].reset_index(drop=True)
            external = frame[frame["split"] == "external"].reset_index(drop=True)
            thresholds = _best_state_thresholds(search, "state_true")
            rng = np.random.default_rng(scenario["seed"])
            # fixed baseline: threshold tuned on validation with a common grid
            val = frame[frame["split"] == "validation"]
            fixed_t, best = float(NU_GRID[0]), -1e9
            for t in NU_GRID:
                a = (val["p"].to_numpy(float) >= t).astype(float)
                s = summarize_group_metrics(grouped_metrics(val, a, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap))
                if s["mean_nu_lb"] > best:
                    best, fixed_t = s["mean_nu_lb"], float(t)
            fixed_nu = summarize_group_metrics(
                grouped_metrics(external, (external["p"].to_numpy(float) >= fixed_t).astype(float),
                                transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
            )["mean_nu_lb"]
            states = external["state_true"].to_numpy(int)
            placebo_rows.append({
                "scenario": name, "seed": scenario["seed"],
                "real_gain": _gain(external, thresholds, states, fixed_nu, bootstrap),
                "circular_shift_gain": _gain(external, thresholds, _circular_shift(states, rng), fixed_nu, bootstrap),
                "block_permute_gain": _gain(external, thresholds, _block_permute(states, rng), fixed_nu, bootstrap),
            })

        # Stability: bootstrap-resample the search split with different seeds.
        frame0 = generate_synthetic_scenario(base)
        frame0["state_true"] = true_states(frame0)
        search0 = frame0[frame0["split"] == "search"].reset_index(drop=True)
        grid = np.linspace(0.30, 0.80, int(drrl_config.get("grid_size", 25)))
        features = []
        freqs = []
        for sd in range(args.stability_seeds):
            cfg = dict(drrl_config)
            cfg["stability_seed"] = 20260928 + sd
            policy = fit_stable_split_policy(search0, grid=grid, config=cfg)
            features.append(policy.feature if policy is not None else "none")
            freqs.append(policy.selection_frequency if policy is not None else 0.0)
        dominant = max(set(features), key=features.count)
        stability_rows.append({
            "scenario": name,
            "stability_rate": float(np.mean([f != "none" for f in features])),
            "dominant_feature": dominant,
            "dominant_feature_rate": float(np.mean([f == dominant for f in features])),
            "mean_selection_frequency": float(np.mean(freqs)),
        })

    placebo = pd.DataFrame(placebo_rows)
    placebo.to_csv(outdir / "stability_placebo.csv", index=False)
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(outdir / "split_stability.csv", index=False)

    summary = {"placebo_by_scenario": [], "stability_by_scenario": stability.to_dict(orient="records")}
    for name, sub in placebo.groupby("scenario"):
        summary["placebo_by_scenario"].append({
            "scenario": name,
            "mean_real_gain": float(sub["real_gain"].mean()),
            "mean_circular_shift_gain": float(sub["circular_shift_gain"].mean()),
            "mean_block_permute_gain": float(sub["block_permute_gain"].mean()),
            "placebo_gap_circular": float((sub["real_gain"] - sub["circular_shift_gain"]).mean()),
            "placebo_gap_block": float((sub["real_gain"] - sub["block_permute_gain"]).mean()),
        })
    (outdir / "stability_placebo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("Placebo (real vs destroyed state alignment):")
    print(pd.DataFrame(summary["placebo_by_scenario"]).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\nSplit stability:")
    print(stability.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
