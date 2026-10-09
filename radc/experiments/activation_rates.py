#!/usr/bin/env python3
"""Activation-rate estimates with exact counts and Wilson 95% intervals (RADC).

Reports, for every controlled setting, the number of activations over the number of
runs and a Wilson score interval, so the activation/abstention claims carry
uncertainty rather than bare fractions. Reads existing result CSVs; no new assets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
Z = 1.959963984540054


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + Z * Z / n
    center = (p + Z * Z / (2.0 * n)) / denom
    half = (Z * math.sqrt(p * (1.0 - p) / n + Z * Z / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _row(group: str, label: str, k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {
        "group": group, "label": label, "activated": int(k), "runs": int(n),
        "rate": (k / n) if n else float("nan"),
        "wilson_lo": lo, "wilson_hi": hi,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(ROOT / "artifacts" / "results"))
    args = parser.parse_args()
    outdir = Path(args.outdir)
    rows: list[dict] = []

    # Experiment C: activation per scenario.
    oracle = pd.read_csv(outdir / "oracle_upper_bound.csv")
    syn = oracle[oracle["dataset"] == "synthetic"]
    for name, sub in syn.groupby("scenario"):
        rows.append(_row("oracle_upper_bound", name, int(sub["drrl_activated"].sum()), len(sub)))
    pos = syn[syn["oracle_lb_gain"] > 0.05]
    neg = syn[syn["oracle_lb_gain"] <= 0.05]
    rows.append(_row("oracle_upper_bound", "positive_controls_pooled", int(pos["drrl_activated"].sum()), len(pos)))
    rows.append(_row("oracle_upper_bound", "negative_controls_pooled", int(neg["drrl_activated"].sum()), len(neg)))
    near = syn[syn["near_zero_oracle"]]
    rows.append(_row("oracle_upper_bound", "abstain_when_near_zero", int((~near["drrl_activated"]).sum()), len(near)))

    # Experiment D: split stability.
    stab = pd.read_csv(outdir / "split_stability.csv")
    for _, r in stab.iterrows():
        k = int(round(r["stability_rate"] * 10))
        rows.append(_row("split_stability", r["scenario"], k, 10))

    # Experiment B: DRRL activation under raw and calibrated scores.
    cd = pd.read_csv(outdir / "calibration_decision_2x2.csv")
    syn2 = cd[cd["dataset"] == "synthetic"]
    for name, sub in syn2[syn2["prob_cal"] == "calibrated"].groupby("scenario"):
        drrl = sub[sub["decision"] == "drrl"]
        rows.append(_row("decision_calibration_activation", name, int(drrl["activated"].sum()), len(drrl)))

    # Strong-control DRRL run (20 seeds).
    strong = outdir / "drrl_strong_control_20seed.csv"
    if strong.exists():
        s = pd.read_csv(strong)
        s = s[s["scenario"] == "decision_relevant_state"]
        rows.append(_row("drrl_strong_control_20seed", "decision_relevant_state", int(s["activated"].sum()), len(s)))

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "activation_rates.csv", index=False)
    pd.set_option("display.width", 160)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Emit a markdown table for direct inclusion.
    lines = ["| Group | Case | Activated/Runs | Rate | Wilson 95% CI |", "|---|---|---:|---:|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['group']} | {r['label']} | {r['activated']}/{r['runs']} | {r['rate']*100:.1f}% | [{r['wilson_lo']*100:.1f}%, {r['wilson_hi']*100:.1f}%] |")
    (outdir / "activation_rates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nwrote", outdir / "activation_rates.csv", "and activation_rates.md")


if __name__ == "__main__":
    main()
