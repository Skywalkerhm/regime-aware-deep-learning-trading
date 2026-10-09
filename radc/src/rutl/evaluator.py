"""Machine-gradeable evaluator for RUTL candidates."""

from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import evaluate_actions
from .data import FEATURE_COLUMNS, feature_matrix
from .metrics import block_bootstrap_lower_bound, block_bootstrap_paired_interval

_BASELINE_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, float], dict[str, float]]] = {}

FORBIDDEN_TOKENS = (
    "subprocess",
    "requests",
    "urllib",
    "socket",
    "open(",
    "Path(",
    "read_csv(",
    "read_pickle(",
)


def _load_candidate(candidate_path: Path):
    text = candidate_path.read_text(encoding="utf-8")
    for token in FORBIDDEN_TOKENS:
        if token in text:
            raise ValueError(f"candidate contains forbidden token: {token}")
    spec = importlib.util.spec_from_file_location("rutl_candidate", candidate_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate from {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy_cls = getattr(module, "ThresholdPolicy", None)
    if policy_cls is None:
        raise AttributeError("candidate must define ThresholdPolicy")
    return policy_cls


def _frame_for_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    part = df[df["split"] == split].copy()
    if part.empty:
        raise ValueError(f"bundle has no rows for split={split!r}")
    return part.sort_values(["asset", "model", "date", "fold"]).reset_index(drop=True)


def _data_dict(df: pd.DataFrame) -> dict[str, np.ndarray]:
    features = feature_matrix(df)
    features["y"] = df["y"].to_numpy(dtype=float)
    features["r_next"] = df["r_next"].to_numpy(dtype=float)
    return features


def _metrics_for_actions(
    df: pd.DataFrame,
    actions: np.ndarray,
    *,
    transaction_cost: float,
    drawdown_penalty: float,
    bootstrap: dict[str, Any],
) -> dict[str, float]:
    result = evaluate_actions(
        actions,
        df["r_next"].to_numpy(dtype=float),
        transaction_cost=transaction_cost,
        drawdown_penalty=drawdown_penalty,
    )
    lower = block_bootstrap_lower_bound(
        result.net_utility,
        block_days=int(bootstrap["block_days"]),
        replications=int(bootstrap["replications"]),
        lower_quantile=float(bootstrap["lower_quantile"]),
        seed=int(bootstrap["seed"]),
    )
    return {
        "nu_lb": lower,
        "mean_net_utility": result.annualized_mean,
        "annualized_vol": result.annualized_vol,
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "turnover": result.turnover,
        "trades": float(result.trades),
    }


def _fixed_threshold_actions(df: pd.DataFrame, threshold: float) -> np.ndarray:
    return (df["u"].to_numpy(dtype=float) >= threshold).astype(float)


def _best_fixed_threshold(
    validation: pd.DataFrame,
    *,
    transaction_cost: float,
    drawdown_penalty: float,
    bootstrap: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_score = -math.inf
    for threshold in np.linspace(0.30, 0.80, 51):
        actions = _fixed_threshold_actions(validation, float(threshold))
        metrics = _metrics_for_actions(
            validation,
            actions,
            transaction_cost=transaction_cost,
            drawdown_penalty=drawdown_penalty,
            bootstrap=bootstrap,
        )
        if metrics["nu_lb"] > best_score:
            best_score = metrics["nu_lb"]
            best_threshold = float(threshold)
    return best_threshold, _metrics_for_actions(
        validation,
        _fixed_threshold_actions(validation, best_threshold),
        transaction_cost=transaction_cost,
        drawdown_penalty=drawdown_penalty,
        bootstrap=bootstrap,
    )


def _current_rule_actions(df: pd.DataFrame) -> np.ndarray:
    thresholds = {
        "crisis": 0.80,
        "trend_up": 0.40,
        "trend_down": 0.50,
        "chop": 0.55,
    }
    regimes = df[["posterior_crisis", "posterior_trend_up", "posterior_trend_down", "posterior_chop"]]
    regime_names = np.array(["crisis", "trend_up", "trend_down", "chop"])
    labels = regime_names[np.argmax(regimes.to_numpy(), axis=1)]
    threshold = np.array([thresholds[label] for label in labels], dtype=float)
    return (df["p"].to_numpy(dtype=float) >= threshold).astype(float)


def _behavior(q: np.ndarray, df: pd.DataFrame) -> dict[str, Any]:
    q = np.asarray(q, dtype=float)
    crisis = df["posterior_crisis"].to_numpy(dtype=float)
    corr = float(np.corrcoef(q, crisis)[0, 1]) if np.std(q) > 1e-12 and np.std(crisis) > 1e-12 else 0.0
    swing = float(np.mean(np.abs(np.diff(q)))) if len(q) > 1 else 0.0
    return {
        "policy_family": "evolved",
        "context_dependence": corr,
        "threshold_swing": swing,
        "turnover_regime": "high" if swing > 0.05 else "medium" if swing > 0.01 else "low",
        "uncertainty_usage": "reported",
    }


def evaluate_candidate(
    candidate_path: Path,
    bundle_path: Path,
    protocol_path: Path,
    *,
    include_external: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    bundle = pd.read_csv(bundle_path, parse_dates=["date"], low_memory=False)
    search = _frame_for_split(bundle, "search")
    validation = _frame_for_split(bundle, "validation")
    external_all = _frame_for_split(bundle, "external")
    external = external_all[external_all["asset"].isin(["BTC-USD", "TSLA"])].copy()
    if external.empty:
        external = external_all.copy()

    try:
        policy_cls = _load_candidate(candidate_path)
        policy = policy_cls()
        policy.fit(_data_dict(search), _data_dict(validation), protocol)
        q = np.asarray(policy.predict_threshold(feature_matrix(validation)), dtype=float)
        if q.shape != (len(validation),):
            raise ValueError(f"predict_threshold returned shape {q.shape}, expected {(len(validation),)}")
        if not np.isfinite(q).all():
            raise ValueError("predict_threshold returned NaN or infinite values")
        q = np.clip(q, *protocol["action_space"]["threshold_bounds"])
        actions = (validation["u"].to_numpy(dtype=float) >= q).astype(float)

        transaction_cost = float(protocol["costs"]["transaction_cost_bps"]) / 10000.0
        drawdown_penalty = float(protocol["utility"]["drawdown_penalty"])
        bootstrap = protocol["primary_metric"]["bootstrap"]
        metrics = _metrics_for_actions(
            validation,
            actions,
            transaction_cost=transaction_cost,
            drawdown_penalty=drawdown_penalty,
            bootstrap=bootstrap,
        )
        cache_key = (
            str(bundle_path.resolve()),
            bundle_path.stat().st_mtime_ns,
            bundle_path.stat().st_size,
            str(protocol_path.resolve()),
        )
        if cache_key not in _BASELINE_CACHE:
            fixed_threshold, fixed_metrics = _best_fixed_threshold(
                validation,
                transaction_cost=transaction_cost,
                drawdown_penalty=drawdown_penalty,
                bootstrap=bootstrap,
            )
            current_actions = _current_rule_actions(validation)
            current_metrics = _metrics_for_actions(
                validation,
                current_actions,
                transaction_cost=transaction_cost,
                drawdown_penalty=drawdown_penalty,
                bootstrap=bootstrap,
            )
            _BASELINE_CACHE[cache_key] = (fixed_threshold, fixed_metrics, current_metrics)
        else:
            fixed_threshold, fixed_metrics, current_metrics = _BASELINE_CACHE[cache_key]
        guardrails = protocol["guardrails"]
        drawdown_cap = max(
            guardrails["max_drawdown_ratio_vs_best_baseline"] * fixed_metrics["max_drawdown"],
            0.35,
        )
        turnover_cap = max(
            guardrails["turnover_ratio_vs_current_rule"] * current_metrics["turnover"],
            guardrails.get("minimum_turnover_cap", 200.0),
        )
        valid = (
            metrics["max_drawdown"] <= drawdown_cap
            and metrics["turnover"] <= turnover_cap
        )
        scores = {
            "nu_lb": metrics["nu_lb"],
            "mean_net_utility": metrics["mean_net_utility"],
            "delta_vs_best_fixed": metrics["nu_lb"] - fixed_metrics["nu_lb"],
            "delta_vs_current_rule": metrics["nu_lb"] - current_metrics["nu_lb"],
            "max_drawdown": metrics["max_drawdown"],
            "turnover": metrics["turnover"],
            "runtime_s": time.perf_counter() - started,
            "correctness": 1.0 if valid else 0.0,
        }
        output: dict[str, Any] = {
            "valid": bool(valid),
            "scores": scores,
            "behavior": _behavior(q, validation),
            "feedback": {
                "fixed_threshold": fixed_threshold,
                "best_fixed_nu_lb": fixed_metrics["nu_lb"],
                "current_rule_nu_lb": current_metrics["nu_lb"],
            },
            "artifacts": {
                "candidate_path": str(candidate_path),
                "bundle_path": str(bundle_path),
                "protocol_path": str(protocol_path),
            },
        }
        if include_external:
            external_q = np.asarray(policy.predict_threshold(feature_matrix(external)), dtype=float)
            external_q = np.clip(external_q, *protocol["action_space"]["threshold_bounds"])
            external_actions = (external["u"].to_numpy(dtype=float) >= external_q).astype(float)
            external_metrics = _metrics_for_actions(
                external,
                external_actions,
                transaction_cost=transaction_cost,
                drawdown_penalty=drawdown_penalty,
                bootstrap=bootstrap,
            )
            external_fixed_metrics = _metrics_for_actions(
                external,
                _fixed_threshold_actions(external, fixed_threshold),
                transaction_cost=transaction_cost,
                drawdown_penalty=drawdown_penalty,
                bootstrap=bootstrap,
            )
            external_current_metrics = _metrics_for_actions(
                external,
                _current_rule_actions(external),
                transaction_cost=transaction_cost,
                drawdown_penalty=drawdown_penalty,
                bootstrap=bootstrap,
            )
            external_metrics["delta_vs_best_fixed"] = external_metrics["nu_lb"] - external_fixed_metrics["nu_lb"]
            external_metrics["delta_vs_current_rule"] = external_metrics["nu_lb"] - external_current_metrics["nu_lb"]
            external_metrics["best_fixed_nu_lb"] = external_fixed_metrics["nu_lb"]
            external_metrics["current_rule_nu_lb"] = external_current_metrics["nu_lb"]
            candidate_net = evaluate_actions(
                external_actions,
                external["r_next"].to_numpy(dtype=float),
                transaction_cost=transaction_cost,
                drawdown_penalty=drawdown_penalty,
            ).net_utility
            fixed_net = evaluate_actions(
                _fixed_threshold_actions(external, fixed_threshold),
                external["r_next"].to_numpy(dtype=float),
                transaction_cost=transaction_cost,
                drawdown_penalty=drawdown_penalty,
            ).net_utility
            paired = candidate_net - fixed_net
            low, high, p_value = block_bootstrap_paired_interval(
                paired,
                block_days=int(bootstrap["block_days"]),
                replications=int(bootstrap["replications"]),
                lower_quantile=float(bootstrap["lower_quantile"]),
                upper_quantile=0.95,
                seed=int(bootstrap["seed"]),
            )
            external_metrics["delta_vs_best_fixed_ci_low"] = low
            external_metrics["delta_vs_best_fixed_ci_high"] = high
            external_metrics["delta_vs_best_fixed_boot_p"] = p_value
            external_metrics["external_guardrails_pass"] = (
                external_metrics["max_drawdown"] <= 1.20 * max(external_fixed_metrics["max_drawdown"], 0.35)
                and external_metrics["turnover"] <= max(1.50 * external_current_metrics["turnover"], 200.0)
            )
            output["external"] = external_metrics
        return output
    except Exception as exc:
        return {
            "valid": False,
            "scores": {
                "nu_lb": -1_000_000.0,
                "mean_net_utility": -1_000_000.0,
                "delta_vs_best_fixed": -1_000_000.0,
                "delta_vs_current_rule": -1_000_000.0,
                "max_drawdown": 1_000_000.0,
                "turnover": 1_000_000.0,
                "runtime_s": time.perf_counter() - started,
                "correctness": 0.0,
            },
            "behavior": {"policy_family": "invalid"},
            "feedback": {"error": f"{type(exc).__name__}: {exc}"},
            "artifacts": {},
        }
