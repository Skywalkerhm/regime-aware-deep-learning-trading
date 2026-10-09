"""Candidate policy interface for the RUTL AlphaEvolve run.

Only the code inside EVOLVE-BLOCK-START/END may be changed by AlphaEvolve.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class ThresholdPolicy:
    """Deterministic linear regime-aware threshold constructor."""

    def __init__(self) -> None:
        self.params: np.ndarray | None = None
        self.bounds: tuple[float, float] = (0.05, 0.95)

    # EVOLVE-BLOCK-START
    def _q(self, features: Mapping[str, np.ndarray], params: np.ndarray) -> np.ndarray:
        q = (
            params[0]
            + params[1] * (features["u"] - 0.5)
            + params[2] * features["crisis"]
            + params[3] * features["trend_up"]
            + params[4] * features["trend_down"]
            + params[5] * features["uncertainty"]
            + params[6] * features["trend_strength"]
        )
        return np.clip(q, *self.bounds)

    @staticmethod
    def _score(actions: np.ndarray, returns: np.ndarray) -> tuple[float, float, float]:
        actions = np.asarray(actions, dtype=float)
        returns = np.asarray(returns, dtype=float)
        changes = np.abs(np.diff(np.r_[0.0, actions]))
        gross = actions * returns - 0.001 * changes
        equity = np.cumprod(1.0 + gross)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (peaks - equity) / np.maximum(peaks, 1e-12)
        mean = float(np.mean(gross) * 252.0)
        vol = float(np.std(gross) * np.sqrt(252.0))
        return mean, vol, float(np.max(drawdowns))

    def fit(
        self,
        train_data: Mapping[str, np.ndarray],
        valid_data: Mapping[str, np.ndarray],
        config: Mapping[str, Any],
    ) -> None:
        rng = np.random.default_rng(int(config.get("random_seed", 20260928)))
        # Search a compact, interpretable policy family. This is intentionally
        # deterministic and cheap enough for repeated AlphaEvolve evaluation.
        lower = np.array([0.20, -0.50, -0.40, -0.30, -0.30, -0.40, -0.30])
        upper = np.array([0.80, 0.50, 0.40, 0.30, 0.30, 0.40, 0.30])
        best_score = -np.inf
        best = np.array([0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        n_candidates = 320
        for _ in range(n_candidates):
            params = rng.uniform(lower, upper)
            q_train = self._q(train_data, params)
            q_valid = self._q(valid_data, params)
            train_actions = (train_data["u"] >= q_train).astype(float)
            valid_actions = (valid_data["u"] >= q_valid).astype(float)
            train_mean, train_vol, train_dd = self._score(train_actions, train_data["r_next"])
            valid_mean, valid_vol, valid_dd = self._score(valid_actions, valid_data["r_next"])
            score = (
                0.35 * train_mean
                + 0.65 * valid_mean
                - 0.05 * valid_vol
                - 0.10 * valid_dd
            )
            if score > best_score:
                best_score = score
                best = params
        self.params = best

    def predict_threshold(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("fit must be called before predict_threshold")
        return self._q(features, self.params)
    # EVOLVE-BLOCK-END

    def state_dict(self) -> dict[str, Any]:
        return {
            "params": None if self.params is None else self.params.tolist(),
            "bounds": list(self.bounds),
        }
