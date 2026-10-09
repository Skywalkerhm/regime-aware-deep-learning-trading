#!/usr/bin/env python3
"""Bin-count sensitivity for the decision-regret mechanism (Experiment A robustness).

Recomputes the oracle regret reduction for the probability-bin count in {5, 10, 20}
and correlates it with the realized external gain (which does not depend on the bin
count). Confirms that the headline correlation is not an artifact of a 10-bin choice.
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
    default_score_edges,
    regret_of_threshold,
    regret_of_threshold_map,
    select_fixed_threshold_by_regret,
    select_state_thresholds_by_regret,
    true_states,
)
from src.rutl.group_metrics import grouped_metrics, select_fixed_threshold_grouped, summarize_group_metrics

SCORE_GRID = np.linspace(0.05, 0.95, 46)
NU_GRID = np.linspace(0.30, 0.80, 51)


def _oracle_regret_reduction(external, states, edges) -> float:
    from src.rutl.decision_regret import build_delta_table
    table = build_delta_table(external, states, edges=edges)
    fixed = select_fixed_threshold_by_regret(external, table, edges=edges, grid=SCORE_GRID, states=states)
    smap = select_state_thresholds_by_regret(external, table, edges=edges, grid=SCORE_GRID, states=states)
    return (regret_of_threshold(external, fixed, table, edges=edges, state_override=states)
            - regret_of_threshold_map(external, smap, table, edges=edges, state_override=states))


def _realized_gain(search, validation, external, state_col, bootstrap) -> float:
    def best_state(part):
        out = {}
        for st in sorted(part[state_col].unique()):
            blk = part[part[state_col] == st]
            bt, bs = float(NU_GRID[0]), -1e9
            for t in NU_GRID:
                a = (blk["p"].to_numpy(float) >= t).astype(float)
                r = evaluate_actions(a, blk["r_next"].to_numpy(float), transaction_cost=0.001, drawdown_penalty=0.25)
                sc = r.annualized_mean - 0.05 * r.annualized_vol - 0.10 * r.max_drawdown
                if sc > bs:
                    bs, bt = sc, t
            out[int(st)] = bt
        return out
    ft, _, _ = select_fixed_threshold_grouped(validation, grid=NU_GRID, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap, score_column="p")
    fixed = summarize_group_metrics(grouped_metrics(external, (external["p"].to_numpy(float) >= ft).astype(float), transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap))["mean_nu_lb"]
    th = best_state(search)
    states = external[state_col].to_numpy(int)
    default = float(np.mean(list(th.values())))
    cut = np.array([th.get(int(s), default) for s in states], float)
    state = summarize_group_metrics(grouped_metrics(external, (external["p"].to_numpy(float) >= cut).astype(float), transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap))["mean_nu_lb"]
    return state - fixed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "data_splits.json"))
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "evaluation_protocol.json"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--outdir", default=str(ROOT / "artifacts" / "results"))
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    bootstrap = protocol["primary_metric"]["bootstrap"]
    scenarios = {s["name"]: s for s in json.loads(Path(args.config).read_text(encoding="utf-8"))["synthetic_scenarios"]}
    rows = []
    for name, base in scenarios.items():
        for off in range(args.repeats):
            sc = dict(base); sc["seed"] = int(base["seed"]) + off * 101
            fr = generate_synthetic_scenario(sc)
            fr["state_true"] = true_states(fr)
            search = fr[fr["split"] == "search"].reset_index(drop=True)
            val = fr[fr["split"] == "validation"].reset_index(drop=True)
            ext = fr[fr["split"] == "external"].reset_index(drop=True)
            gain = _realized_gain(search, val, ext, "state_true", bootstrap)
            for nb in [5, 10, 20]:
                red = _oracle_regret_reduction(ext, ext["state_true"].to_numpy(int), default_score_edges(nb))
                rows.append({"scenario": name, "seed": sc["seed"], "n_bins": nb, "regret_reduction": red, "realized_gain": gain})
    df = pd.DataFrame(rows)
    out = Path(args.outdir) / "bin_count_sensitivity.csv"
    df.to_csv(out, index=False)
    summary = []
    for nb in [5, 10, 20]:
        sub = df[df["n_bins"] == nb]
        r = float(np.corrcoef(sub["regret_reduction"], sub["realized_gain"])[0, 1]) if sub["regret_reduction"].std() > 1e-12 else float("nan")
        summary.append({"n_bins": nb, "corr_regret_reduction__realized_gain": r, "n": len(sub)})
    pd.DataFrame(summary).to_csv(Path(args.outdir) / "bin_count_sensitivity_summary.csv", index=False)
    pd.set_option("display.width", 160)
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
