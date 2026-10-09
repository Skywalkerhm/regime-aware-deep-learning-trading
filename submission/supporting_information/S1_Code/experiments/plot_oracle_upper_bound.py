#!/usr/bin/env python3
"""Oracle upper-bound figure for the RADC mechanism (Experiment C)."""

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


def main() -> None:
    df = pd.read_csv(ROOT / "artifacts" / "results" / "oracle_upper_bound.csv")
    syn = df[df["dataset"] == "synthetic"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    colors = {"decision_relevant_state": "#1f77b4", "asymmetric_utility": "#ff7f0e"}
    for name, sub in syn.groupby("scenario"):
        positive = sub["oracle_lb_gain"] > 0.05
        ax.scatter(sub["oracle_lb_gain"], sub["drrl_delta"],
                   s=26, alpha=0.7, label=name, color=colors.get(name, "#7f7f7f"))
    lim = [min(-0.05, syn["oracle_lb_gain"].min()), max(1.3, syn["oracle_lb_gain"].max())]
    ax.plot(lim, lim, "k--", linewidth=1, label="y = x (perfect recovery)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0.05, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("oracle lower-bound gain (true states)")
    ax.set_ylabel("DRRL-VoI realized gain")
    ax.set_title("Recovery of the oracle upper bound")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1]
    summary = pd.DataFrame(json.loads((ROOT / "artifacts" / "results" / "oracle_upper_bound_summary.json").read_text())["by_scenario"])
    x = np.arange(len(summary))
    ax.bar(x - 0.22, summary["mean_oracle_lb_gain"], width=0.44, label="oracle LB gain")
    ax.bar(x + 0.22, summary["mean_drrl_delta"], width=0.44, label="DRRL delta")
    ax.axhline(0.05, color="grey", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["scenario"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("external NU_LB gain")
    ax.set_title("Oracle ceiling vs captured gain")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = ROOT / "artifacts" / "reports" / "figures" / "oracle_upper_bound.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
