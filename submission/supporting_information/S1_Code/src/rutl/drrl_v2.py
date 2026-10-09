"""DRRL-VoI v0.2: utility-sensitive depth-2 tree and risk-controlled gate."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .backtest import evaluate_actions
from .drrl import CONTEXT_FEATURES, _best_threshold, _utility_proxy
from .group_metrics import GROUP_COLUMNS, grouped_metrics, select_fixed_threshold_grouped, summarize_group_metrics


def _leaf(frame: pd.DataFrame, grid: np.ndarray, config: dict) -> dict[str, Any]:
    threshold, _ = _best_threshold(
        frame,
        grid,
        vol_penalty=float(config.get("vol_penalty", 0.05)),
        dd_penalty=float(config.get("dd_penalty", 0.10)),
    )
    return {"type": "leaf", "threshold": threshold}


def apply_tree(frame: pd.DataFrame, tree: dict[str, Any] | float) -> np.ndarray:
    if isinstance(tree, (float, int, np.floating)):
        return (frame["p"].to_numpy(dtype=float) >= float(tree)).astype(float)
    if tree["type"] == "leaf":
        return (frame["p"].to_numpy(dtype=float) >= float(tree["threshold"])).astype(float)
    low = frame[tree["feature"]].to_numpy(dtype=float) <= float(tree["cutoff"])
    output = np.zeros(len(frame), dtype=float)
    output[low] = apply_tree(frame.loc[low], tree["left"])
    output[~low] = apply_tree(frame.loc[~low], tree["right"])
    return output


def _tree_score(frame: pd.DataFrame, tree: dict[str, Any], config: dict) -> float:
    actions = apply_tree(frame, tree)
    return _utility_proxy(
        frame,
        actions,
        vol_penalty=float(config.get("vol_penalty", 0.05)),
        dd_penalty=float(config.get("dd_penalty", 0.10)),
    )


def _n_splits(tree: dict[str, Any]) -> int:
    if tree["type"] == "leaf":
        return 0
    return 1 + _n_splits(tree["left"]) + _n_splits(tree["right"])


def fit_tree(frame: pd.DataFrame, *, grid: np.ndarray, config: dict, depth: int = 0) -> dict[str, Any]:
    max_depth = int(config.get("max_depth", 2))
    min_leaf = int(config.get("min_leaf", 40))
    features = config.get("features") or CONTEXT_FEATURES
    quantiles = tuple(config.get("quantiles", (0.33, 0.5, 0.67)))
    complexity_penalty = float(config.get("complexity_penalty", 0.01))
    base = _leaf(frame, grid, config)
    best_tree = base
    best_score = _tree_score(frame, base, config)
    if depth >= max_depth or len(frame) < 2 * min_leaf:
        return best_tree
    for feature in features:
        values = frame[feature].to_numpy(dtype=float)
        for quantile in quantiles:
            cutoff = float(np.quantile(values, quantile))
            low = frame[frame[feature] <= cutoff]
            high = frame[frame[feature] > cutoff]
            if len(low) < min_leaf or len(high) < min_leaf:
                continue
            left = fit_tree(low, grid=grid, config=config, depth=depth + 1)
            right = fit_tree(high, grid=grid, config=config, depth=depth + 1)
            candidate = {"type": "split", "feature": feature, "cutoff": cutoff, "left": left, "right": right}
            score = _tree_score(frame, candidate, config) - complexity_penalty * _n_splits(candidate)
            if score > best_score:
                best_score = score
                best_tree = candidate
    return best_tree



def _validation_group_differences_v2(frame: pd.DataFrame, tree: dict[str, Any], fixed: float) -> np.ndarray:
    work = frame.copy()
    work["_tree"] = apply_tree(frame, tree)
    work["_fixed"] = apply_tree(frame, fixed)
    differences: list[float] = []
    groups = list(work.groupby(GROUP_COLUMNS, sort=True))
    if len(groups) == 1:
        ordered = groups[0][1].sort_values("date").reset_index(drop=True)
        midpoint = len(ordered) // 2
        groups = [("early", ordered.iloc[:midpoint]), ("late", ordered.iloc[midpoint:])]
    for _, group in groups:
        group = group.sort_values("date").reset_index(drop=True)
        a = evaluate_actions(group["_tree"].to_numpy(dtype=float), group["r_next"].to_numpy(dtype=float), transaction_cost=0.001, drawdown_penalty=0.25)
        b = evaluate_actions(group["_fixed"].to_numpy(dtype=float), group["r_next"].to_numpy(dtype=float), transaction_cost=0.001, drawdown_penalty=0.25)
        differences.append(float(a.net_utility.mean() * 252.0 - b.net_utility.mean() * 252.0))
    return np.asarray(differences, dtype=float)


def _risk_ok(candidate: dict[str, float], fixed: dict[str, float], config: dict) -> bool:
    dd_tol = float(config.get("dd_tolerance", 0.10))
    turnover_tol = float(config.get("turnover_tolerance", 0.25))
    return (
        candidate["mean_max_drawdown"] <= (1.0 + dd_tol) * fixed["mean_max_drawdown"]
        and candidate["mean_turnover"] <= (1.0 + turnover_tol) * fixed["mean_turnover"]
    )


def evaluate_drrl_v2(search: pd.DataFrame, validation: pd.DataFrame, external: pd.DataFrame, *, protocol: dict, config: dict | None = None) -> dict:
    config = config or {}
    grid = np.linspace(0.30, 0.80, int(config.get("grid_size", 31)))
    bootstrap = protocol["primary_metric"]["bootstrap"]
    fixed_threshold, _, _ = select_fixed_threshold_grouped(
        validation,
        grid=grid,
        transaction_cost=0.001,
        drawdown_penalty=0.25,
        bootstrap=bootstrap,
        score_column="p",
    )
    search_tree = fit_tree(search, grid=grid, config=config)
    validation_diffs = _validation_group_differences_v2(validation, search_tree, fixed_threshold)
    rng = np.random.default_rng(int(bootstrap["seed"]))
    samples = rng.choice(validation_diffs, size=(int(bootstrap["replications"]), len(validation_diffs)), replace=True).mean(axis=1)
    voi_low = float(np.quantile(samples, float(bootstrap["lower_quantile"])))
    positive_fraction = float(np.mean(validation_diffs > 0.0))
    validation_tree_metrics = summarize_group_metrics(
        grouped_metrics(validation, apply_tree(validation, search_tree), transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
    )
    validation_fixed_metrics = summarize_group_metrics(
        grouped_metrics(validation, apply_tree(validation, fixed_threshold), transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
    )
    activated = bool(
        voi_low > float(config.get("activation_margin", 0.05))
        and positive_fraction >= float(config.get("min_positive_fraction", 0.9))
        and _risk_ok(validation_tree_metrics, validation_fixed_metrics, config)
    )
    selected: dict[str, Any] | float = fixed_threshold
    if activated:
        combined = pd.concat([search, validation], ignore_index=True)
        selected = fit_tree(combined, grid=grid, config=config)
    policy_metrics = summarize_group_metrics(
        grouped_metrics(external, apply_tree(external, selected), transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
    )
    fixed_metrics = summarize_group_metrics(
        grouped_metrics(external, apply_tree(external, fixed_threshold), transaction_cost=0.001, drawdown_penalty=0.25, bootstrap=bootstrap)
    )
    return {
        "fixed_threshold": fixed_threshold,
        "tree": search_tree,
        "n_splits": _n_splits(search_tree),
        "voi_low": voi_low,
        "positive_fraction": positive_fraction,
        "activated": activated,
        "external_policy": policy_metrics,
        "external_fixed": fixed_metrics,
        "external_delta": policy_metrics["mean_nu_lb"] - fixed_metrics["mean_nu_lb"],
    }
