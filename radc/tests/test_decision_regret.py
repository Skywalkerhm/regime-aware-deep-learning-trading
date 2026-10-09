from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rutl.data import generate_synthetic_scenario
from src.rutl.decision_regret import (
    _bin_index,
    brier_score,
    build_delta_table,
    decision_regret,
    default_score_edges,
    expected_calibration_error,
    regret_of_threshold,
    regret_of_threshold_map,
    select_fixed_threshold_by_regret,
    select_state_thresholds_by_regret,
    true_states,
)

EDGES = default_score_edges(10)
GRID = np.linspace(0.05, 0.95, 46)


def _regret_reduction(frame: pd.DataFrame, seed: int) -> float:
    rng = np.random.default_rng(seed)
    external = frame[frame["split"] == "external"].reset_index(drop=True)
    states = true_states(external)
    table = build_delta_table(external, states, edges=EDGES)
    fixed = select_fixed_threshold_by_regret(external, table, edges=EDGES, grid=GRID, states=states)
    state_map = select_state_thresholds_by_regret(external, table, edges=EDGES, grid=GRID, states=states)
    r_fixed = regret_of_threshold(external, fixed, table, edges=EDGES, state_override=states)
    r_state = regret_of_threshold_map(external, state_map, table, edges=EDGES, state_override=states)
    return r_fixed - r_state


def _scenario(name: str, seed_offset: int = 0) -> pd.DataFrame:
    manifest = json.loads((ROOT / "configs" / "data_splits.json").read_text(encoding="utf-8"))
    base = {s["name"]: s for s in manifest["synthetic_scenarios"]}[name]
    scenario = dict(base)
    scenario["seed"] = int(base["seed"]) + seed_offset * 101
    return generate_synthetic_scenario(scenario)


class DecisionRegretTest(unittest.TestCase):
    def test_bayes_action_has_zero_regret(self) -> None:
        frame = _scenario("decision_relevant_state")
        external = frame[frame["split"] == "external"].reset_index(drop=True)
        states = true_states(external)
        table = build_delta_table(external, states, edges=EDGES)
        bins = _bin_index(external["p"].to_numpy(dtype=float), EDGES)
        delta = np.array([table.lookup(int(b), int(s)) for b, s in zip(bins, states)])
        bayes_actions = (delta >= 0.0).astype(float)
        self.assertAlmostEqual(decision_regret(bins, states, bayes_actions, table), 0.0, places=10)

    def test_positive_control_has_positive_regret_reduction(self) -> None:
        reductions = [_regret_reduction(_scenario("decision_relevant_state", k), 1 + k) for k in range(3)]
        self.assertTrue(np.mean(reductions) > 1e-3, reductions)

    def test_uninformative_state_has_no_regret_reduction(self) -> None:
        reductions = [_regret_reduction(_scenario("uninformative_regime", k), 1 + k) for k in range(3)]
        self.assertTrue(abs(np.mean(reductions)) < 1e-3, reductions)

    def test_statistical_separation_without_decision_value_gives_no_gain(self) -> None:
        # high_separation has well-separated predictive distributions but a
        # state-invariant payoff, so it must not yield a decision gain.
        reductions = [_regret_reduction(_scenario("high_separation", k), 1 + k) for k in range(3)]
        self.assertTrue(abs(np.mean(reductions)) < 1e-3, reductions)

    def test_state_shuffle_destroys_regret_reduction(self) -> None:
        frame = _scenario("decision_relevant_state")
        external = frame[frame["split"] == "external"].reset_index(drop=True)
        states = true_states(external)
        table = build_delta_table(external, states, edges=EDGES)
        fixed = select_fixed_threshold_by_regret(external, table, edges=EDGES, grid=GRID, states=states)
        state_map = select_state_thresholds_by_regret(external, table, edges=EDGES, grid=GRID, states=states)
        real_reduction = (
            regret_of_threshold(external, fixed, table, edges=EDGES, state_override=states)
            - regret_of_threshold_map(external, state_map, table, edges=EDGES, state_override=states)
        )
        shuffled_table = build_delta_table(external, np.random.default_rng(0).permutation(states), edges=EDGES)
        shuffled_reduction = (
            regret_of_threshold(external, fixed, shuffled_table, edges=EDGES, state_override=states)
            - regret_of_threshold_map(external, state_map, shuffled_table, edges=EDGES, state_override=states)
        )
        self.assertTrue(real_reduction > shuffled_reduction)

    def test_ece_does_not_separate_decision_relevant_from_irrelevant(self) -> None:
        def ece_for(name: str, offset: int) -> float:
            frame = _scenario(name, offset)
            external = frame[frame["split"] == "external"].reset_index(drop=True)
            return expected_calibration_error(external["p"].to_numpy(float), external["y"].to_numpy(float))

        pos = np.mean([ece_for("decision_relevant_state", k) for k in range(3)])
        neg = np.mean([ece_for("high_separation", k) for k in range(3)])
        # Both are reasonably calibrated probabilities, so ECE is small in both and
        # cannot be used to tell decision-relevant from decision-irrelevant states.
        self.assertLess(pos, 0.06)
        self.assertLess(neg, 0.06)
        self.assertLess(abs(pos - neg), 0.05)

    def test_brier_is_bounded(self) -> None:
        frame = _scenario("asymmetric_utility")
        external = frame[frame["split"] == "external"].reset_index(drop=True)
        value = brier_score(external["p"].to_numpy(float), external["y"].to_numpy(float))
        self.assertGreater(value, 0.0)
        self.assertLess(value, 0.5)


if __name__ == "__main__":
    unittest.main()
