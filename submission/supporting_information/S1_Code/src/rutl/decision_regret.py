"""Decision-regret and value-of-information diagnostics for regime gating.

This module operationalizes the *decision-calibration* (not probability
calibration) view of regime-aware thresholding.

For a decision at observation ``t`` define the counterfactual action values

    V1(p, z) = E[U(a=1) | P=p, Z=z]
    V0(p, z) = E[U(a=0) | P=p, Z=z]
    Delta(p, z) = V1(p, z) - V0(p, z)

with the gross long/flat utility ``U(1) = r_next`` and ``U(0) = 0``. The Bayes
action is ``a*(p, z) = 1[Delta(p, z) >= 0]``. Decision regret of a threshold rule
``theta`` is the expected utility loss from taking the wrong action:

    R_dec(theta) = E[ max(0, -Delta) * 1[p >= theta(z)]
                    + max(0,  Delta) * 1[p <  theta(z)] ].

Regime-aware decision calibration is confirmed when a state-dependent threshold
strictly reduces ``R_dec`` relative to the best fixed threshold, and when that
regret reduction -- not probability-calibration quality (ECE / Brier) -- predicts
the realized out-of-sample utility gain.

The module is deliberately utility-agnostic about switching and drawdown costs:
those are risk overlays. ``Delta`` here isolates the *decision-relevance* of the
state partition, which is exactly the mechanism the theory is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

POSTERIOR_COLUMNS = [
    "posterior_crisis",
    "posterior_trend_up",
    "posterior_trend_down",
    "posterior_chop",
]
RULE_LABELS = ("crisis", "trend_up", "trend_down", "chop")
RULE_THRESHOLDS = {"crisis": 0.80, "trend_up": 0.40, "trend_down": 0.50, "chop": 0.55}


def default_score_edges(n_bins: int = 10) -> np.ndarray:
    """Bin edges over the probability score in (0, 1)."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    return np.linspace(0.0, 1.0, n_bins + 1)


def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    idx = np.digitize(values, edges) - 1
    return np.clip(idx, 0, len(edges) - 2).astype(int)


def detector_states(frame: pd.DataFrame) -> np.ndarray:
    """Map the soft regime posterior to a hard detector state index (0..3)."""
    posterior = frame[POSTERIOR_COLUMNS].to_numpy(dtype=float)
    return np.argmax(posterior, axis=1).astype(int)


def detector_labels(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(RULE_LABELS, dtype=object)[detector_states(frame)]


def true_states(frame: pd.DataFrame) -> np.ndarray:
    if "truth_regime" not in frame.columns:
        raise KeyError("frame has no truth_regime column")
    return frame["truth_regime"].to_numpy(dtype=int)


@dataclass
class DeltaTable:
    """Conditional expected utility gap Delta(p, z) estimated in score bins."""

    edges: np.ndarray
    cell: dict[tuple[int, int], float] = field(default_factory=dict)
    bin_mean: dict[int, float] = field(default_factory=dict)
    state_mean: dict[int, float] = field(default_factory=dict)
    global_mean: float = 0.0

    def lookup(self, bin_idx: int, state: int) -> float:
        if (bin_idx, state) in self.cell:
            return self.cell[(bin_idx, state)]
        if bin_idx in self.bin_mean:
            return self.bin_mean[bin_idx]
        if state in self.state_mean:
            return self.state_mean[state]
        return self.global_mean


def estimate_delta_table(
    deltas: np.ndarray,
    bins: np.ndarray,
    states: np.ndarray,
    *,
    edges: np.ndarray,
) -> DeltaTable:
    """Estimate Delta(p, z) from realized per-observation utility gaps."""
    deltas = np.asarray(deltas, dtype=float)
    bins = np.asarray(bins, dtype=int)
    states = np.asarray(states, dtype=int)
    if not (len(deltas) == len(bins) == len(states)):
        raise ValueError("deltas, bins, and states must have equal length")

    table = DeltaTable(edges=np.asarray(edges, dtype=float))
    table.global_mean = float(np.mean(deltas)) if len(deltas) else 0.0
    for index in np.unique(bins):
        mask = bins == index
        table.bin_mean[int(index)] = float(np.mean(deltas[mask]))
    for state in np.unique(states):
        mask = states == state
        table.state_mean[int(state)] = float(np.mean(deltas[mask]))
    for index in np.unique(bins):
        for state in np.unique(states):
            mask = (bins == index) & (states == state)
            if np.any(mask):
                table.cell[(int(index), int(state))] = float(np.mean(deltas[mask]))
    return table


def build_delta_table(
    frame: pd.DataFrame,
    states: np.ndarray,
    *,
    edges: np.ndarray,
    return_col: str = "r_next",
) -> DeltaTable:
    deltas = frame[return_col].to_numpy(dtype=float)
    bins = _bin_index(frame["p"].to_numpy(dtype=float), edges)
    return estimate_delta_table(deltas, bins, np.asarray(states, dtype=int), edges=edges)


def decision_regret(
    bins: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    table: DeltaTable,
) -> float:
    """Mean per-observation decision regret of ``actions`` under ``table``."""
    bins = np.asarray(bins, dtype=int)
    states = np.asarray(states, dtype=int)
    actions = np.asarray(actions, dtype=float)
    if not (len(bins) == len(states) == len(actions)):
        raise ValueError("bins, states, and actions must have equal length")
    delta = np.array([table.lookup(int(b), int(s)) for b, s in zip(bins, states)], dtype=float)
    missed_gain = np.maximum(0.0, delta) * (1.0 - actions)  # should hold, went flat
    avoided_gain = np.maximum(0.0, -delta) * actions       # should go flat, held
    return float(np.mean(missed_gain + avoided_gain))


def regret_of_threshold(
    frame: pd.DataFrame,
    threshold: float,
    table: DeltaTable,
    *,
    edges: np.ndarray,
    state_override: np.ndarray | None = None,
) -> float:
    """Regret of a single common threshold ``1[p >= threshold]``."""
    bins = _bin_index(frame["p"].to_numpy(dtype=float), edges)
    states = _resolve_states(frame, state_override)
    actions = (frame["p"].to_numpy(dtype=float) >= float(threshold)).astype(float)
    return decision_regret(bins, states, actions, table)


def regret_of_threshold_map(
    frame: pd.DataFrame,
    thresholds: Mapping[int, float],
    table: DeltaTable,
    *,
    edges: np.ndarray,
    state_override: np.ndarray | None = None,
    fallback: float | None = None,
) -> float:
    """Regret of a state-dependent threshold map ``theta(z)``.

    States present in ``frame`` but absent from ``thresholds`` fall back to
    ``fallback`` (default: the mean of the supplied thresholds). This keeps the
    map well defined when a state is unseen during fitting.
    """
    bins = _bin_index(frame["p"].to_numpy(dtype=float), edges)
    states = _resolve_states(frame, state_override)
    if not thresholds:
        raise ValueError("thresholds map is empty")
    default = float(np.mean(list(thresholds.values()))) if fallback is None else float(fallback)
    p = frame["p"].to_numpy(dtype=float)
    applied = np.array(
        [p[i] >= float(thresholds.get(int(states[i]), default)) for i in range(len(frame))],
        dtype=float,
    )
    return decision_regret(bins, states, applied, table)


def _resolve_states(frame: pd.DataFrame, state_override: np.ndarray | None) -> np.ndarray:
    if state_override is not None:
        return np.asarray(state_override, dtype=int)
    if "state" in frame.columns:
        return frame["state"].to_numpy(dtype=int)
    return detector_states(frame)


def select_fixed_threshold_by_regret(
    train: pd.DataFrame,
    table: DeltaTable,
    *,
    edges: np.ndarray,
    grid: np.ndarray,
    states: np.ndarray | None = None,
) -> float:
    bins = _bin_index(train["p"].to_numpy(dtype=float), edges)
    resolved = _resolve_states(train, states)
    best_threshold = float(grid[0])
    best_regret = np.inf
    for threshold in grid:
        actions = (train["p"].to_numpy(dtype=float) >= float(threshold)).astype(float)
        value = decision_regret(bins, resolved, actions, table)
        if value < best_regret:
            best_regret = value
            best_threshold = float(threshold)
    return best_threshold


def select_state_thresholds_by_regret(
    train: pd.DataFrame,
    table: DeltaTable,
    *,
    edges: np.ndarray,
    grid: np.ndarray,
    states: np.ndarray | None = None,
) -> dict[int, float]:
    bins = _bin_index(train["p"].to_numpy(dtype=float), edges)
    resolved = _resolve_states(train, states)
    p = train["p"].to_numpy(dtype=float)
    out: dict[int, float] = {}
    for state in np.unique(resolved):
        mask = resolved == state
        best_threshold = float(grid[0])
        best_regret = np.inf
        for threshold in grid:
            actions = (p[mask] >= float(threshold)).astype(float)
            value = decision_regret(bins[mask], resolved[mask], actions, table)
            if value < best_regret:
                best_regret = value
                best_threshold = float(threshold)
        out[int(state)] = best_threshold
    return out


def regret_curve(
    frame: pd.DataFrame,
    table: DeltaTable,
    *,
    edges: np.ndarray,
    grid: np.ndarray,
    state_override: np.ndarray | None = None,
) -> pd.DataFrame:
    """Regret as a function of a single common threshold (decision-regret curve)."""
    bins = _bin_index(frame["p"].to_numpy(dtype=float), edges)
    states = _resolve_states(frame, state_override)
    rows = []
    for threshold in grid:
        actions = (frame["p"].to_numpy(dtype=float) >= float(threshold)).astype(float)
        rows.append({"threshold": float(threshold), "regret": decision_regret(bins, states, actions, table)})
    return pd.DataFrame(rows)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> float:
    """Standard equal-width ECE over the forecast probability."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    total = len(p)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        weight = np.sum(mask) / total
        ece += weight * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(ece)


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def rule_thresholds_by_detector(frame: pd.DataFrame) -> np.ndarray:
    labels = detector_labels(frame)
    return np.array([RULE_THRESHOLDS[label] for label in labels], dtype=float)
