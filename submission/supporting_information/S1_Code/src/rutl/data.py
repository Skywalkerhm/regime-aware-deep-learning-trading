"""Data generation and canonical schemas for RUTL."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REGIME_COLUMNS = ["crisis", "trend_up", "trend_down", "chop"]
FEATURE_COLUMNS = ["u", "uncertainty", "volatility_20d", "trend_strength", "return_20d"] + REGIME_COLUMNS


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def soft_regime_posterior(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic soft-rule posterior using established regime features."""
    vol = df["volatility_20d"].to_numpy(dtype=float)
    trend = df["sma_ratio_20_50"].to_numpy(dtype=float) - 1.0
    ret20 = df["return_20d"].to_numpy(dtype=float)

    crisis_score = (
        2.5 * _sigmoid((vol - 0.186) / 0.025)
        + 2.5 * _sigmoid((-ret20 - 0.10) / 0.025)
    )
    trend_up_score = 2.0 * _sigmoid((trend - 0.008) / 0.006)
    trend_down_score = 2.0 * _sigmoid((-trend - 0.008) / 0.006)
    chop_score = 0.7 * np.ones_like(vol)

    scores = np.column_stack([crisis_score, trend_up_score, trend_down_score, chop_score])
    # Keep the canonical ordering: crisis, trend_up, trend_down, chop.
    scores = np.clip(scores, -20.0, 20.0)
    scores -= scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    out = pd.DataFrame(probs, columns=REGIME_COLUMNS, index=df.index)
    out["uncertainty"] = entropy / np.log(len(REGIME_COLUMNS))
    return out


def generate_synthetic_scenario(config: dict[str, Any]) -> pd.DataFrame:
    """Generate a deterministic regime DGP with a held-out tail."""
    rng = np.random.default_rng(int(config["seed"]))
    n = int(config["n_obs"])
    k = int(config["n_regimes"])
    persistence = float(config["persistence"])
    separation = float(config["separation"])
    detector_error = str(config["detector_error"])
    dispersion = str(config["probability_dispersion"])

    regimes = np.zeros(n, dtype=int)
    for t in range(1, n):
        if rng.random() < persistence:
            regimes[t] = regimes[t - 1]
        else:
            regimes[t] = int(rng.integers(0, k))

    state_center = np.linspace(-separation, separation, k) if k > 1 else np.array([0.0])
    state_scale = 0.55 if dispersion == "low" else 1.15
    logits = state_center[regimes] + rng.normal(0.0, state_scale, size=n)
    p = _sigmoid(logits)

    asym = config.get("utility_asymmetry")
    decision_relevance = config.get("decision_relevance")
    if decision_relevance == "strong":
        # State 0 rewards upside and truncates downside; state 1 reverses the asymmetry.
        gain = np.where(regimes == 0, 0.035, 0.005)
        loss = np.where(regimes == 0, 0.005, 0.035)
    elif asym == "strong":
        gain = 0.018 + 0.012 * (regimes == 0)
        loss = 0.014 + 0.012 * (regimes == 1 if k > 1 else 0)
    else:
        gain = np.full(n, 0.012)
        loss = np.full(n, 0.012)
    y = rng.binomial(1, p, size=n)
    r_next = np.where(y == 1, gain, -loss)

    truth = np.eye(k)[regimes]
    true_probs = np.zeros((n, len(REGIME_COLUMNS)), dtype=float)
    true_probs[:, 0] = truth[:, 0] if k > 0 else 1.0
    true_probs[:, 1] = truth[:, 1] if k > 1 else 0.0
    true_probs[:, 2] = truth[:, 2] if k > 2 else 0.0
    true_probs[:, 3] = truth[:, 3] if k > 3 else 0.0

    if detector_error == "uniform_posterior":
        posterior = np.full_like(true_probs, 1.0 / len(REGIME_COLUMNS))
    elif detector_error == "high":
        posterior = 0.55 * true_probs + 0.45 * (1.0 / len(REGIME_COLUMNS))
    else:
        posterior = 0.92 * true_probs + 0.08 * (1.0 / len(REGIME_COLUMNS))
    posterior = posterior / posterior.sum(axis=1, keepdims=True)

    u = np.clip(0.45 * p + 0.55 * (p - p.mean()) / (p.std() + 1e-8) * 0.20 + 0.5, 0.0, 1.0)
    uncertainty = -(posterior * np.log(np.clip(posterior, 1e-12, 1.0))).sum(axis=1) / np.log(len(REGIME_COLUMNS))
    split = np.where(np.arange(n) < int(0.5 * n), "search",
                     np.where(np.arange(n) < int(0.75 * n), "validation", "external"))
    return pd.DataFrame({
        "date": pd.date_range("2000-01-01", periods=n, freq="D"),
        "asset": config["name"],
        "model": "synthetic",
        "fold": "synthetic",
        "split": split,
        "p": p,
        "u": u,
        "y": y,
        "r_next": r_next,
        "regime": regimes,
        "truth_regime": regimes,
        "posterior_crisis": posterior[:, 0],
        "posterior_trend_up": posterior[:, 1],
        "posterior_trend_down": posterior[:, 2],
        "posterior_chop": posterior[:, 3],
        "uncertainty": uncertainty,
        "volatility_20d": 0.03 + 0.05 * (regimes == 0),
        "trend_strength": 0.02 * (regimes == 1) - 0.02 * (regimes == 2),
        "return_20d": 0.05 * (regimes == 1) - 0.05 * (regimes == 2),
    })


def feature_matrix(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return candidate-visible features without returns or labels."""
    return {
        "u": df["u"].to_numpy(dtype=float),
        "uncertainty": df["uncertainty"].to_numpy(dtype=float),
        "volatility_20d": df["volatility_20d"].to_numpy(dtype=float),
        "trend_strength": df["trend_strength"].to_numpy(dtype=float),
        "return_20d": df["return_20d"].to_numpy(dtype=float),
        "crisis": df["posterior_crisis"].to_numpy(dtype=float),
        "trend_up": df["posterior_trend_up"].to_numpy(dtype=float),
        "trend_down": df["posterior_trend_down"].to_numpy(dtype=float),
        "chop": df["posterior_chop"].to_numpy(dtype=float),
    }
