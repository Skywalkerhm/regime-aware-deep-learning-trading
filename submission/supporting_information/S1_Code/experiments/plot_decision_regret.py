#!/usr/bin/env python3
"""Publication figures for the Regime-Aware Decision Calibration mechanism.

Figure 1 (mechanism): conditional action value Delta(p, z) and the decision-regret
curve, contrasting a decision-relevant state DGP with a statistically separated but
decision-irrelevant one.

Figure 2 (confirmation): per-scenario regret reduction vs ECE vs realized gain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rutl.data import generate_synthetic_scenario
from src.rutl.decision_regret import (
    _bin_index,
    build_delta_table,
    default_score_edges,
    regret_of_threshold_map,
    select_state_thresholds_by_regret,
    true_states,
)

EDGES = default_score_edges(10)
CENTER = (EDGES[:-1] + EDGES[1:]) / 2
SEEDS = 12



def _state_aware_regret(name: str, base: dict, grid: np.ndarray) -> float:
    vals = []
    for offset in range(SEEDS):
        scenario = dict(base)
        scenario["seed"] = int(base["seed"]) + offset * 101
        frame = generate_synthetic_scenario(scenario)
        external = frame[frame["split"] == "external"].reset_index(drop=True)
        states = true_states(external)
        table = build_delta_table(external, states, edges=EDGES)
        tmap = select_state_thresholds_by_regret(external, table, edges=EDGES, grid=grid, states=states)
        vals.append(regret_of_threshold_map(external, tmap, table, edges=EDGES, state_override=states))
    return float(np.mean(vals))


def _delta_curve(name: str, base: dict) -> dict[int, np.ndarray]:
    acc: dict[int, list[np.ndarray]] = {}
    for offset in range(SEEDS):
        scenario = dict(base)
        scenario["seed"] = int(base["seed"]) + offset * 101
        frame = generate_synthetic_scenario(scenario)
        external = frame[frame["split"] == "external"].reset_index(drop=True)
        states = true_states(external)
        bins = _bin_index(external["p"].to_numpy(dtype=float), EDGES)
        r = external["r_next"].to_numpy(dtype=float)
        for state in np.unique(states):
            curve = np.full(len(CENTER), np.nan)
            for b in range(len(CENTER)):
                mask = (bins == b) & (states == state)
                if mask.any():
                    curve[b] = r[mask].mean()
            acc.setdefault(int(state), []).append(curve)
    return {s: np.nanmean(np.vstack(v), axis=0) for s, v in acc.items()}


def figure_mechanism(base_pos: dict, base_neg: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    pos = _delta_curve("decision_relevant_state", base_pos)
    neg = _delta_curve("uninformative_regime", base_neg)
    for state, curve in pos.items():
        ax.plot(CENTER, curve, marker="o", label=f"relevant: state {state}")
    for state, curve in neg.items():
        ax.plot(CENTER, curve, linestyle="--", alpha=0.5, label=f"irrelevant: state {state}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("forecast probability  p")
    ax.set_ylabel(r"conditional action value  $\Delta(p,z)$")
    ax.set_title("State changes the optimal action\n(positive) vs state-invariant (negative)")
    ax.legend(fontsize=7, ncol=1)
    ax.grid(alpha=0.25)

    ax = axes[1]
    curves = pd.read_csv(ROOT / "artifacts" / "results" / "decision_regret_curves.csv")
    for name, style in [("decision_relevant_state", "-"), ("uninformative_regime", "--")]:
        sub = curves[curves["scenario"] == name].sort_values("threshold")
        ax.plot(sub["threshold"], sub["regret"], style, marker="o", markersize=3, label=name)
    state_aware = _state_aware_regret("decision_relevant_state", base_pos, np.linspace(0.05, 0.95, 46))
    sub_pos = curves[curves["scenario"] == "decision_relevant_state"].sort_values("threshold")
    fixed_min = float(sub_pos["regret"].min())
    ax.axhline(state_aware, color="#1f77b4", linestyle=":", linewidth=1.5)
    ax.annotate("", xy=(0.5, state_aware), xytext=(0.5, fixed_min),
                arrowprops=dict(arrowstyle="<->", color="#1f77b4"))
    ax.text(0.52, (state_aware + fixed_min) / 2, f"VoI $\\approx$ {fixed_min - state_aware:.4f}", color="#1f77b4", fontsize=8)
    ax.set_xlabel("fixed threshold  $\\theta$")
    ax.set_ylabel(r"decision regret  $R_{dec}(\theta)$")
    ax.set_title("Decision-regret curve\n(gap to state-aware minimum = VoI)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figure_confirmation(summary_path: Path, out: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(summary["by_scenario"])
    rows = rows.sort_values("regret_reduction_oracle", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(rows))
    ax = axes[0]
    ax.scatter(rows["regret_reduction_oracle"], rows["realized_gain"], s=60, c="#1f77b4")
    for _, row in rows.iterrows():
        ax.annotate(row["scenario"], (row["regret_reduction_oracle"], row["realized_gain"]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("oracle regret reduction")
    ax.set_ylabel("realized external gain (NU_LB)")
    ax.set_title(f"regret reduction ~ gain  (r = {summary['corr_regret_reduction_oracle__realized_gain']:+.2f})")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.bar(x - 0.2, rows["regret_reduction_oracle"], width=0.4, label="regret reduction")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, rows["ece"], width=0.4, color="#d62728", label="ECE")
    ax.set_xticks(x)
    ax.set_xticklabels(rows["scenario"], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("regret reduction")
    ax2.set_ylabel("ECE (red)")
    ax.set_title(f"ECE ~ gain  (r = {summary['corr_ece__realized_gain']:+.2f})")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    manifest = json.loads((ROOT / "configs" / "data_splits.json").read_text(encoding="utf-8"))
    scen = {s["name"]: s for s in manifest["synthetic_scenarios"]}
    figdir = ROOT / "artifacts" / "reports" / "figures"
    figure_mechanism(scen["decision_relevant_state"], scen["uninformative_regime"], figdir / "decision_regret_mechanism.png")
    figure_confirmation(ROOT / "artifacts" / "results" / "decision_regret_summary.json", figdir / "decision_regret_confirmation.png")
    print("figures written to", figdir)


if __name__ == "__main__":
    main()
