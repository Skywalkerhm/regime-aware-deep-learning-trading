"""Group-level evaluation utilities for applied asset/model/fold analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .evaluator import _metrics_for_actions


GROUP_COLUMNS = ["asset", "fold", "model"]


def grouped_metrics(
    frame: pd.DataFrame,
    actions: np.ndarray,
    *,
    transaction_cost: float,
    drawdown_penalty: float,
    bootstrap: dict[str, Any],
) -> pd.DataFrame:
    if len(frame) != len(actions):
        raise ValueError("frame and actions must have equal length")
    work = frame.copy()
    work["_action"] = np.asarray(actions, dtype=float)
    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(GROUP_COLUMNS, sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        metrics = _metrics_for_actions(
            group,
            group["_action"].to_numpy(dtype=float),
            transaction_cost=transaction_cost,
            drawdown_penalty=drawdown_penalty,
            bootstrap=bootstrap,
        )
        asset, fold, model = keys
        rows.append({"asset": asset, "fold": fold, "model": model, "n": len(group), **metrics})
    return pd.DataFrame(rows)


def summarize_group_metrics(groups: pd.DataFrame) -> dict[str, float]:
    return {
        "n_groups": float(len(groups)),
        "mean_nu_lb": float(groups["nu_lb"].mean()),
        "median_nu_lb": float(groups["nu_lb"].median()),
        "mean_net_utility": float(groups["mean_net_utility"].mean()),
        "mean_sharpe": float(groups["sharpe"].mean()),
        "mean_max_drawdown": float(groups["max_drawdown"].mean()),
        "mean_turnover": float(groups["turnover"].mean()),
    }


def select_fixed_threshold_grouped(
    validation: pd.DataFrame,
    *,
    grid: np.ndarray,
    transaction_cost: float,
    drawdown_penalty: float,
    bootstrap: dict[str, Any],
    score_column: str = "u",
) -> tuple[float, pd.DataFrame, dict[str, float]]:
    best_threshold = float(grid[0])
    best_summary: dict[str, float] | None = None
    best_groups: pd.DataFrame | None = None
    for threshold in grid:
        actions = (validation[score_column].to_numpy(dtype=float) >= float(threshold)).astype(float)
        groups = grouped_metrics(
            validation,
            actions,
            transaction_cost=transaction_cost,
            drawdown_penalty=drawdown_penalty,
            bootstrap=bootstrap,
        )
        summary = summarize_group_metrics(groups)
        if best_summary is None or summary["mean_nu_lb"] > best_summary["mean_nu_lb"]:
            best_threshold = float(threshold)
            best_summary = summary
            best_groups = groups
    if best_groups is None or best_summary is None:
        raise RuntimeError("threshold selection failed")
    return best_threshold, best_groups, best_summary
