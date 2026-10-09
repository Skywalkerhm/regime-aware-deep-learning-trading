"""Decision-Relevant Regime Learning with Value-of-Information Gating (v0.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import evaluate_actions
from .group_metrics import GROUP_COLUMNS, grouped_metrics, select_fixed_threshold_grouped, summarize_group_metrics
from .metrics import block_bootstrap_paired_interval

CONTEXT_FEATURES = [
    "uncertainty",
    "volatility_20d",
    "trend_strength",
    "posterior_crisis",
    "posterior_trend_up",
    "posterior_trend_down",
    "posterior_chop",
]


@dataclass(frozen=True)
class SplitPolicy:
    feature: str
    cutoff: float
    low_threshold: float
    high_threshold: float
    search_score: float


@dataclass(frozen=True)
class StableSplitPolicy:
    feature: str
    cutoff: float
    low_threshold: float
    high_threshold: float
    search_score: float
    selection_frequency: float
    n_bootstrap: int


def _utility_proxy(frame: pd.DataFrame, actions: np.ndarray, *, vol_penalty: float = 0.05, dd_penalty: float = 0.10) -> float:
    """Fast group-level selection score used before bootstrap validation."""
    groups = frame.copy()
    groups["_action"] = actions
    scores = []
    for _, group in groups.groupby(GROUP_COLUMNS, sort=True):
        group = group.sort_values("date")
        result = evaluate_actions(
            group["_action"].to_numpy(dtype=float),
            group["r_next"].to_numpy(dtype=float),
            transaction_cost=0.001,
            drawdown_penalty=0.25,
        )
        scores.append(result.annualized_mean - vol_penalty * result.annualized_vol - dd_penalty * result.max_drawdown)
    return float(np.mean(scores)) if scores else -1e9


def _best_threshold(frame: pd.DataFrame, grid: np.ndarray, *, vol_penalty: float = 0.05, dd_penalty: float = 0.10) -> tuple[float, float]:
    best_threshold = float(grid[0])
    best_score = -1e9
    for threshold in grid:
        actions = (frame["p"].to_numpy(dtype=float) >= float(threshold)).astype(float)
        score = _utility_proxy(frame, actions, vol_penalty=vol_penalty, dd_penalty=dd_penalty)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def fit_split_policy(
    search: pd.DataFrame,
    *,
    grid: np.ndarray,
    complexity_penalty: float = 0.01,
    features: list[str] | None = None,
    quantiles: tuple[float, ...] = (0.33, 0.50, 0.67),
    vol_penalty: float = 0.05,
    dd_penalty: float = 0.10,
) -> SplitPolicy | None:
    """Fit a one-level decision-relevant state partition on search data."""
    best: SplitPolicy | None = None
    for feature in (features or CONTEXT_FEATURES):
        values = search[feature].to_numpy(dtype=float)
        for quantile in quantiles:
            cutoff = float(np.quantile(values, quantile))
            low = search[search[feature] <= cutoff]
            high = search[search[feature] > cutoff]
            if len(low) < 20 or len(high) < 20:
                continue
            low_threshold, low_score = _best_threshold(low, grid, vol_penalty=vol_penalty, dd_penalty=dd_penalty)
            high_threshold, high_score = _best_threshold(high, grid, vol_penalty=vol_penalty, dd_penalty=dd_penalty)
            actions = np.where(
                search[feature].to_numpy(dtype=float) <= cutoff,
                search["p"].to_numpy(dtype=float) >= low_threshold,
                search["p"].to_numpy(dtype=float) >= high_threshold,
            ).astype(float)
            score = _utility_proxy(search, actions, vol_penalty=vol_penalty, dd_penalty=dd_penalty) - complexity_penalty
            candidate = SplitPolicy(feature, cutoff, low_threshold, high_threshold, score)
            if best is None or candidate.search_score > best.search_score:
                best = candidate
    return best


def fit_stable_split_policy(
    search: pd.DataFrame,
    *,
    grid: np.ndarray,
    config: dict,
) -> StableSplitPolicy | None:
    """Select a one-level split that survives group/time bootstrap resampling."""
    n_bootstrap = int(config.get("n_bootstrap", 40))
    stability_threshold = float(config.get("stability_threshold", 0.55))
    rng = np.random.default_rng(int(config.get("stability_seed", 20260928)))
    policies: list[SplitPolicy] = []
    groups = list(search.groupby(GROUP_COLUMNS, sort=False))
    for _ in range(n_bootstrap):
        if len(groups) > 1:
            chosen = rng.integers(0, len(groups), size=len(groups))
            sample = pd.concat([groups[i][1] for i in chosen], ignore_index=True)
        else:
            # Block bootstrap for a single asset/model/fold time series.
            ordered = groups[0][1].sort_values("date").reset_index(drop=True)
            block = int(config.get("bootstrap_block", 50))
            blocks = max(1, int(np.ceil(len(ordered) / block)))
            starts = rng.integers(0, max(1, len(ordered) - block + 1), size=blocks)
            pieces = [ordered.iloc[start:start + block] for start in starts]
            sample = pd.concat(pieces, ignore_index=True)
        policy = fit_split_policy(
            sample,
            grid=grid,
            complexity_penalty=float(config.get("complexity_penalty", 0.01)),
            features=config.get("features"),
            quantiles=tuple(config.get("quantiles", (0.33, 0.50, 0.67))),
            vol_penalty=float(config.get("vol_penalty", 0.05)),
            dd_penalty=float(config.get("dd_penalty", 0.10)),
        )
        if policy is not None:
            policies.append(policy)
    if not policies:
        return None
    counts: dict[str, int] = {}
    for policy in policies:
        counts[policy.feature] = counts.get(policy.feature, 0) + 1
    feature = max(counts, key=counts.get)
    selected = [policy for policy in policies if policy.feature == feature]
    cutoffs = np.asarray([policy.cutoff for policy in selected], dtype=float)
    median_cutoff = float(np.median(cutoffs))
    spread = max(0.01, float(np.quantile(cutoffs, 0.75) - np.quantile(cutoffs, 0.25)))
    cluster = [policy for policy in selected if abs(policy.cutoff - median_cutoff) <= max(0.02, spread)]
    frequency = len(cluster) / n_bootstrap
    if frequency < stability_threshold:
        return None
    low = float(np.median([policy.low_threshold for policy in cluster]))
    high = float(np.median([policy.high_threshold for policy in cluster]))
    score = float(np.mean([policy.search_score for policy in cluster]))
    return StableSplitPolicy(feature, median_cutoff, low, high, score, frequency, n_bootstrap)


def apply_policy(frame: pd.DataFrame, policy: SplitPolicy | StableSplitPolicy | float) -> np.ndarray:
    if isinstance(policy, (float, int, np.floating)):
        return (frame["p"].to_numpy(dtype=float) >= float(policy)).astype(float)
    low = frame[policy.feature].to_numpy(dtype=float) <= policy.cutoff
    p = frame["p"].to_numpy(dtype=float)
    return np.where(low, p >= policy.low_threshold, p >= policy.high_threshold).astype(float)


def _validation_group_differences(
    frame: pd.DataFrame,
    policy_a: SplitPolicy | float,
    policy_b: SplitPolicy | float,
) -> np.ndarray:
    work = frame.copy()
    work["_a"] = apply_policy(frame, policy_a)
    work["_b"] = apply_policy(frame, policy_b)
    differences: list[float] = []
    groups = list(work.groupby(GROUP_COLUMNS, sort=True))
    if len(groups) == 1:
        # Use two contiguous time blocks when only one asset/model/fold group exists.
        ordered = groups[0][1].sort_values("date").reset_index(drop=True)
        midpoint = len(ordered) // 2
        groups = [("early", ordered.iloc[:midpoint]), ("late", ordered.iloc[midpoint:])]
    for _, group in groups:
        group = group.sort_values("date").reset_index(drop=True)
        a = evaluate_actions(group["_a"].to_numpy(dtype=float), group["r_next"].to_numpy(dtype=float), transaction_cost=0.001, drawdown_penalty=0.25)
        b = evaluate_actions(group["_b"].to_numpy(dtype=float), group["r_next"].to_numpy(dtype=float), transaction_cost=0.001, drawdown_penalty=0.25)
        differences.append(float(a.net_utility.mean() * 252.0 - b.net_utility.mean() * 252.0))
    return np.asarray(differences, dtype=float)


def _voi_gate(
    differences: np.ndarray,
    *,
    bootstrap: dict,
    activation_margin: float,
    min_positive_fraction: float,
) -> dict[str, float | bool]:
    rng = np.random.default_rng(int(bootstrap["seed"]))
    values = np.asarray(differences, dtype=float)
    samples = rng.choice(values, size=(int(bootstrap["replications"]), len(values)), replace=True).mean(axis=1)
    lower = float(np.quantile(samples, float(bootstrap["lower_quantile"])))
    upper = float(np.quantile(samples, 0.95))
    positive_fraction = float(np.mean(values > 0.0))
    mean_value = float(np.mean(values))
    return {
        "voi_low": lower,
        "voi_high": upper,
        "mean_voi": mean_value,
        "p_voi": float((np.sum(samples <= 0.0) + 1.0) / (len(samples) + 1.0)),
        "positive_fraction": positive_fraction,
        "activated": bool(lower > activation_margin and positive_fraction >= min_positive_fraction),
    }

def _conformal_risk_buffer(
    validation: pd.DataFrame,
    candidate_policy: SplitPolicy,
    fixed_threshold: float,
    *,
    protocol: dict,
    config: dict,
) -> float:
    bootstrap = protocol["primary_metric"]["bootstrap"]
    candidate = grouped_metrics(
        validation,
        apply_policy(validation, candidate_policy),
        transaction_cost=0.001,
        drawdown_penalty=0.25,
        bootstrap=bootstrap,
    )[["asset", "fold", "model", "max_drawdown", "turnover"]]
    fixed = grouped_metrics(
        validation,
        apply_policy(validation, fixed_threshold),
        transaction_cost=0.001,
        drawdown_penalty=0.25,
        bootstrap=bootstrap,
    )[["asset", "fold", "model", "max_drawdown", "turnover"]]
    merged = candidate.merge(fixed, on=["asset", "fold", "model"], suffixes=("_candidate", "_fixed"))
    dd_increase = np.maximum(0.0, merged["max_drawdown_candidate"] - merged["max_drawdown_fixed"])
    turnover_increase = np.maximum(0.0, merged["turnover_candidate"] - merged["turnover_fixed"]) / 252.0
    risk = dd_increase.to_numpy(dtype=float) + float(config.get("turnover_risk_weight", 0.25)) * turnover_increase.to_numpy(dtype=float)
    alpha = float(config.get("conformal_alpha", 0.10))
    # Split-conformal style finite-sample upper quantile.
    n = len(risk)
    rank = min(n, int(np.ceil((n + 1) * (1.0 - alpha))))
    return float(np.sort(risk)[rank - 1]) if n else float("inf")


def evaluate_drrl(
    search: pd.DataFrame,
    validation: pd.DataFrame,
    external: pd.DataFrame,
    *,
    protocol: dict,
    config: dict | None = None,
    grid: np.ndarray | None = None,
) -> dict:
    config = config or {}
    if grid is None:
        grid_size = int(config.get("grid_size", 51))
        grid = np.linspace(0.30, 0.80, grid_size)
    else:
        grid = np.asarray(grid, dtype=float)
    fixed_threshold, _, _ = select_fixed_threshold_grouped(
        validation,
        grid=grid,
        transaction_cost=0.001,
        drawdown_penalty=0.25,
        bootstrap=protocol["primary_metric"]["bootstrap"],
        score_column="p",
    )
    if config.get("stability_selection", False):
        split_policy = fit_stable_split_policy(search, grid=grid, config=config)
    else:
        split_policy = fit_split_policy(
            search,
            grid=grid,
            complexity_penalty=float(config.get("complexity_penalty", 0.01)),
            features=config.get("features"),
            quantiles=tuple(config.get("quantiles", (0.33, 0.50, 0.67))),
            vol_penalty=float(config.get("vol_penalty", 0.05)),
            dd_penalty=float(config.get("dd_penalty", 0.10)),
        )
    bootstrap = protocol["primary_metric"]["bootstrap"]
    if split_policy is None:
        fixed_metrics = summarize_group_metrics(
            grouped_metrics(
                external,
                apply_policy(external, fixed_threshold),
                transaction_cost=0.001,
                drawdown_penalty=0.25,
                bootstrap=bootstrap,
            )
        )
        return {
            "fixed_threshold": fixed_threshold,
            "split_policy": None,
            "voi_low": float("-inf"),
            "voi_high": float("-inf"),
            "mean_voi": float("nan"),
            "p_voi": float("nan"),
            "activated": False,
            "external_policy": fixed_metrics,
            "external_fixed": fixed_metrics,
            "external_delta": 0.0,
        }
    validation_differences = _validation_group_differences(validation, split_policy, fixed_threshold)
    risk_buffer = _conformal_risk_buffer(
        validation,
        split_policy,
        fixed_threshold,
        protocol=protocol,
        config=config,
    )
    gate = _voi_gate(
        validation_differences,
        bootstrap=bootstrap,
        activation_margin=float(config.get("activation_margin", protocol.get("drrl", {}).get("activation_margin", 0.05))),
        min_positive_fraction=float(config.get("min_positive_fraction", protocol.get("drrl", {}).get("min_positive_fraction", 0.90))),
    )
    gate["activated"] = bool(gate["activated"] and float(gate["voi_low"]) > risk_buffer)
    gate["risk_buffer"] = risk_buffer
    voi_low = float(gate["voi_low"])
    voi_high = float(gate["voi_high"])
    mean_voi = float(gate["mean_voi"])
    p_voi = float(gate["p_voi"])
    activated = bool(gate["activated"])
    if activated:
        combined = pd.concat([search, validation], ignore_index=True)
        if config.get("stability_selection", False):
            selected_policy = fit_stable_split_policy(combined, grid=grid, config=config) or fixed_threshold
        else:
            selected_policy = fit_split_policy(
                combined,
                grid=grid,
                complexity_penalty=float(config.get("complexity_penalty", 0.01)),
                features=config.get("features"),
                quantiles=tuple(config.get("quantiles", (0.33, 0.50, 0.67))),
                vol_penalty=float(config.get("vol_penalty", 0.05)),
                dd_penalty=float(config.get("dd_penalty", 0.10)),
            ) or fixed_threshold
    else:
        selected_policy = fixed_threshold

    external_metrics = summarize_group_metrics(
        grouped_metrics(
            external,
            apply_policy(external, selected_policy),
            transaction_cost=0.001,
            drawdown_penalty=0.25,
            bootstrap=bootstrap,
        )
    )
    fixed_metrics = summarize_group_metrics(
        grouped_metrics(
            external,
            apply_policy(external, fixed_threshold),
            transaction_cost=0.001,
            drawdown_penalty=0.25,
            bootstrap=bootstrap,
        )
    )
    return {
        "fixed_threshold": fixed_threshold,
        "split_policy": {
            "feature": split_policy.feature,
            "cutoff": split_policy.cutoff,
            "low_threshold": split_policy.low_threshold,
            "high_threshold": split_policy.high_threshold,
            "search_score": split_policy.search_score,
            "selection_frequency": getattr(split_policy, "selection_frequency", 1.0),
        },
        "voi_low": voi_low,
        "voi_high": voi_high,
        "mean_voi": mean_voi,
        "p_voi": p_voi,
        "positive_fraction": gate["positive_fraction"],
        "risk_buffer": gate.get("risk_buffer", 0.0),
        "activated": activated,
        "external_policy": external_metrics,
        "external_fixed": fixed_metrics,
        "external_delta": external_metrics["mean_nu_lb"] - fixed_metrics["mean_nu_lb"],
    }
