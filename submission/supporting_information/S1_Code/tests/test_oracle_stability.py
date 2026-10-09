from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.stability_placebo import (
    _actions,
    _best_state_thresholds,
    _block_permute,
    _gain,
)
from src.rutl.data import generate_synthetic_scenario
from src.rutl.decision_regret import true_states
from src.rutl.drrl import _voi_gate
from src.rutl.group_metrics import grouped_metrics, summarize_group_metrics

NU_GRID = np.linspace(0.30, 0.80, 51)


def _frame(name: str):
    manifest = json.loads((ROOT / "configs" / "data_splits.json").read_text(encoding="utf-8"))
    base = {s["name"]: s for s in manifest["synthetic_scenarios"]}[name]
    frame = generate_synthetic_scenario(base)
    frame["state_true"] = true_states(frame)
    return frame


class OracleStabilityTest(unittest.TestCase):
    def test_placebo_destroys_decision_relevant_gain(self) -> None:
        protocol = json.loads((ROOT / "configs" / "evaluation_protocol.json").read_text(encoding="utf-8"))
        bootstrap = protocol["primary_metric"]["bootstrap"]
        frame = _frame("decision_relevant_state")
        search = frame[frame["split"] == "search"]
        validation = frame[frame["split"] == "validation"]
        external = frame[frame["split"] == "external"].reset_index(drop=True)
        thresholds = _best_state_thresholds(search, "state_true")

        best_t, best = float(NU_GRID[0]), -1e9
        for t in NU_GRID:
            a = (validation["p"].to_numpy(float) >= t).astype(float)
            s = summarize_group_metrics(grouped_metrics(validation, a, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap))
            if s["mean_nu_lb"] > best:
                best, best_t = s["mean_nu_lb"], float(t)
        fixed_nu = summarize_group_metrics(
            grouped_metrics(external, (external["p"].to_numpy(float) >= best_t).astype(float),
                            transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
        )["mean_nu_lb"]

        states = external["state_true"].to_numpy(int)
        real_gain = _gain(external, thresholds, states, fixed_nu, bootstrap)
        rng = np.random.default_rng(20260928)
        placebo_gain = _gain(external, thresholds, _block_permute(states, rng), fixed_nu, bootstrap)
        self.assertGreater(real_gain, 0.5)
        self.assertLess(placebo_gain, 0.5)
        self.assertGreater(real_gain - placebo_gain, 0.5)

    def test_voi_gate_rejects_zero_value(self) -> None:
        gate = _voi_gate(np.zeros(6), bootstrap={"replications": 500, "lower_quantile": 0.05, "seed": 1}, activation_margin=0.05, min_positive_fraction=0.9)
        self.assertFalse(gate["activated"])

    def test_voi_gate_accepts_strong_positive_value(self) -> None:
        gate = _voi_gate(np.array([0.4, 0.5, 0.6, 0.55, 0.45, 0.5]), bootstrap={"replications": 500, "lower_quantile": 0.05, "seed": 1}, activation_margin=0.05, min_positive_fraction=0.9)
        self.assertTrue(gate["activated"])

    def test_statistically_separated_but_irrelevant_state_has_no_gain(self) -> None:
        protocol = json.loads((ROOT / "configs" / "evaluation_protocol.json").read_text(encoding="utf-8"))
        bootstrap = protocol["primary_metric"]["bootstrap"]
        frame = _frame("high_separation")
        search = frame[frame["split"] == "search"]
        validation = frame[frame["split"] == "validation"]
        external = frame[frame["split"] == "external"].reset_index(drop=True)
        thresholds = _best_state_thresholds(search, "state_true")

        best_t, best = float(NU_GRID[0]), -1e9
        for t in NU_GRID:
            a = (validation["p"].to_numpy(float) >= t).astype(float)
            s = summarize_group_metrics(grouped_metrics(validation, a, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap))
            if s["mean_nu_lb"] > best:
                best, best_t = s["mean_nu_lb"], float(t)
        fixed_nu = summarize_group_metrics(
            grouped_metrics(external, (external["p"].to_numpy(float) >= best_t).astype(float),
                            transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
        )["mean_nu_lb"]
        states = external["state_true"].to_numpy(int)
        # State-dependent gain is ~0 despite statistical separation of the states.
        gain = _gain(external, thresholds, states, fixed_nu, bootstrap)
        self.assertLess(abs(gain), 0.2)


if __name__ == "__main__":
    unittest.main()
