"""Deterministic metrics for RUTL evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UtilityResult:
    net_utility: np.ndarray
    total_return: float
    annualized_mean: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    turnover: float
    trades: int


def max_drawdown(returns: np.ndarray) -> float:
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdown = (equity - peaks) / np.maximum(peaks, 1e-12)
    return float(-np.min(drawdown)) if len(drawdown) else 0.0


def simulate_utility(
    actions: np.ndarray,
    returns: np.ndarray,
    *,
    transaction_cost: float = 0.001,
    drawdown_penalty: float = 0.25,
) -> UtilityResult:
    """Evaluate long/flat actions against next-period returns."""
    actions = np.asarray(actions, dtype=float)
    returns = np.asarray(returns, dtype=float)
    if actions.shape != returns.shape:
        raise ValueError(f"actions and returns must have equal shape: {actions.shape} != {returns.shape}")
    if len(actions) == 0:
        raise ValueError("empty action/return arrays")

    preview = np.r_[0.0, actions]
    changes = np.abs(np.diff(preview))
    gross = actions * returns - transaction_cost * changes
    equity_path = np.cumprod(1.0 + gross)
    peaks = np.maximum.accumulate(equity_path)
    drawdowns = (peaks - equity_path) / np.maximum(peaks, 1e-12)
    prior_max_dd = np.r_[0.0, np.maximum.accumulate(drawdowns)[:-1]]
    dd_increments = np.maximum(drawdowns - prior_max_dd, 0.0)
    net = gross - drawdown_penalty * dd_increments
    equity = float(equity_path[-1])
    max_dd = float(np.max(drawdowns))
    trades = int(np.count_nonzero(changes))
    prev = 0.0
    ann_mean = float(np.mean(net) * 252.0)
    ann_vol = float(np.std(net) * np.sqrt(252.0))
    sharpe = ann_mean / (ann_vol + 1e-12)
    return UtilityResult(
        net_utility=net,
        total_return=float(equity - 1.0),
        annualized_mean=ann_mean,
        annualized_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        turnover=float(np.mean(np.abs(np.diff(np.r_[0.0, actions]))) * 252.0),
        trades=trades,
    )


def block_bootstrap_lower_bound(
    net_utility: np.ndarray,
    *,
    block_days: int = 10,
    replications: int = 2000,
    lower_quantile: float = 0.05,
    seed: int = 20260928,
) -> float:
    """Lower quantile of the moving-block bootstrap mean annualized utility."""
    values = np.asarray(net_utility, dtype=float)
    if len(values) == 0:
        raise ValueError("net_utility is empty")
    block_days = max(1, min(int(block_days), len(values)))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(values) / block_days))
    max_start = max(1, len(values) - block_days + 1)
    offsets = np.arange(block_days)
    samples = np.empty(replications, dtype=float)
    cursor = 0
    chunk_size = 128
    while cursor < replications:
        size = min(chunk_size, replications - cursor)
        starts = rng.integers(0, max_start, size=(size, n_blocks))
        gathered = values[starts[:, :, None] + offsets]
        sampled = gathered.reshape(size, -1)[:, :len(values)]
        samples[cursor:cursor + size] = sampled.mean(axis=1) * 252.0
        cursor += size
    return float(np.quantile(samples, lower_quantile))

def block_bootstrap_paired_interval(
    differences: np.ndarray,
    *,
    block_days: int = 10,
    replications: int = 2000,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    seed: int = 20260928,
) -> tuple[float, float, float]:
    """Moving-block bootstrap interval for a paired mean difference."""
    values = np.asarray(differences, dtype=float)
    if len(values) == 0:
        raise ValueError("differences is empty")
    block_days = max(1, min(int(block_days), len(values)))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(values) / block_days))
    max_start = max(1, len(values) - block_days + 1)
    offsets = np.arange(block_days)
    samples = np.empty(replications, dtype=float)
    cursor = 0
    chunk_size = 128
    while cursor < replications:
        size = min(chunk_size, replications - cursor)
        starts = rng.integers(0, max_start, size=(size, n_blocks))
        gathered = values[starts[:, :, None] + offsets]
        sampled = gathered.reshape(size, -1)[:, :len(values)]
        samples[cursor:cursor + size] = sampled.mean(axis=1) * 252.0
        cursor += size
    p_value = float((np.sum(samples <= 0.0) + 1.0) / (replications + 1.0))
    return (
        float(np.quantile(samples, lower_quantile)),
        float(np.quantile(samples, upper_quantile)),
        p_value,
    )
