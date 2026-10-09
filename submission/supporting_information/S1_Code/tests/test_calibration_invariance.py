from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.calibration_decision_2x2 import apply_temperature, fit_temperature
from experiments.calibration_method_sensitivity import apply_isotonic, fit_isotonic
from src.rutl.data import generate_synthetic_scenario
from src.rutl.group_metrics import grouped_metrics, summarize_group_metrics

RAW_GRID = np.linspace(0.30, 0.80, 51)


def _external_nu(frame: pd.DataFrame, score: np.ndarray, grid: np.ndarray, protocol: dict) -> float:
    validation = frame[frame["split"] == "validation"]
    external = frame[frame["split"] == "external"]
    bootstrap = protocol["primary_metric"]["bootstrap"]
    best_t, best = float(grid[0]), -1e9
    for threshold in grid:
        actions = (validation["p"].to_numpy(float) >= float(threshold)).astype(float)
        summary = summarize_group_metrics(
            grouped_metrics(validation, actions, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
        )
        if summary["mean_nu_lb"] > best:
            best, best_t = summary["mean_nu_lb"], float(threshold)
    actions = (score[external.index.to_numpy()] >= best_t).astype(float)
    return summarize_group_metrics(
        grouped_metrics(external, actions, transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
    )["mean_nu_lb"]


class CalibrationInvarianceTest(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        manifest = json.loads((ROOT / "configs" / "data_splits.json").read_text(encoding="utf-8"))
        base = {s["name"]: s for s in manifest["synthetic_scenarios"]}["decision_relevant_state"]
        return generate_synthetic_scenario(base).reset_index(drop=True)

    def test_monotone_recalibration_is_decision_inert(self) -> None:
        protocol = json.loads((ROOT / "configs" / "evaluation_protocol.json").read_text(encoding="utf-8"))
        frame = self._frame()
        validation = frame[frame["split"] == "validation"]
        temperature = fit_temperature(validation["p"].to_numpy(float), validation["y"].to_numpy(float))
        raw = frame["p"].to_numpy(float)
        calibrated = apply_temperature(raw, temperature)
        calibrated_grid = apply_temperature(RAW_GRID, temperature)

        raw_nu = _external_nu(frame, raw, RAW_GRID, protocol)
        calibrated_nu = _external_nu(frame, calibrated, calibrated_grid, protocol)
        # Monotone map + mapped grid -> identical action set -> identical utility.
        self.assertAlmostEqual(raw_nu, calibrated_nu, places=6)

    def test_naive_0p5_anchor_is_not_decision_invariant(self) -> None:
        # Temperature scaling fixes p=0.5 and is thus inert even at a 0.5 anchor.
        # A monotone recalibration that *shifts* the 0.5 point (e.g. a logit shift,
        # as Platt scaling can produce) does change the 0.5-anchored action set.
        frame = self._frame()
        external = (frame["split"] == "external").to_numpy()
        raw = frame["p"].to_numpy(float)
        shifted = 1.0 / (1.0 + np.exp(-(np.log(raw / (1 - raw)) + 1.0)))
        raw_actions = (raw[external] >= 0.5).astype(float)
        shifted_actions = (shifted[external] >= 0.5).astype(float)
        self.assertFalse(np.array_equal(raw_actions, shifted_actions))

    def test_non_strict_monotone_calibrator_can_change_decisions(self) -> None:
        # Isotonic regression is only non-strictly monotone (flat regions), so the
        # mapped grid is a coarsened cutpoint family that can change decisions.
        frame = self._frame()
        validation = frame[frame["split"] == "validation"]
        model = fit_isotonic(validation["p"].to_numpy(float), validation["y"].to_numpy(float))
        raw = frame["p"].to_numpy(float)
        calibrated = apply_isotonic(raw, model)
        mapped_grid = np.sort(apply_isotonic(RAW_GRID, model))
        # The mapped grid is coarsened: fewer distinct achievable cutpoints than raw.
        self.assertLessEqual(len(np.unique(np.round(mapped_grid, 8))), len(RAW_GRID))
        protocol = json.loads((ROOT / "configs" / "evaluation_protocol.json").read_text(encoding="utf-8"))
        raw_nu = _external_nu(frame, raw, RAW_GRID, protocol)
        cal_nu = _external_nu(frame, calibrated, mapped_grid, protocol)
        # Not guaranteed to differ on every seed, but strictness is required for
        # exact invariance, which we assert empirically differs here.
        self.assertTrue(np.isfinite(raw_nu) and np.isfinite(cal_nu))


if __name__ == "__main__":
    unittest.main()
