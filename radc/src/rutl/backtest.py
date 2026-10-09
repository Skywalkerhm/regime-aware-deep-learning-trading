"""Backtest facade for RUTL."""

from __future__ import annotations

import numpy as np

from .metrics import UtilityResult, simulate_utility


def evaluate_actions(
    actions: np.ndarray,
    returns: np.ndarray,
    *,
    transaction_cost: float = 0.001,
    drawdown_penalty: float = 0.25,
) -> UtilityResult:
    return simulate_utility(
        actions,
        returns,
        transaction_cost=transaction_cost,
        drawdown_penalty=drawdown_penalty,
    )
