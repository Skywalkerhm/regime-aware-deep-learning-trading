#!/usr/bin/env python3
"""AlphaEvolve-style runner for RUTL v1.

The LLM-level strategy is encoded in a small set of executable policy
families. Each generation samples an elite parent from the AlphaEvolve
database, generates breadth/depth/repair candidates, evaluates them with the
frozen evaluator, and archives every result.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from string import Template

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path("/Users/mh/.codex/skills/alpha-evolve/scripts/evolve_db.py")
sys.path.insert(0, str(ROOT))

from src.rutl.evaluator import evaluate_candidate

BUNDLE = ROOT / "artifacts" / "bundles" / "predictions_v1.csv"
PROTOCOL = ROOT / "configs" / "evaluation_protocol.json"

FAMILIES = {
    "linear": [
        ("u_centered", '(features["u"] - 0.5)', -0.50, 0.50),
        ("crisis", 'features["crisis"]', -0.50, 0.50),
        ("trend_up", 'features["trend_up"]', -0.40, 0.40),
        ("trend_down", 'features["trend_down"]', -0.40, 0.40),
        ("uncertainty", 'features["uncertainty"]', -0.50, 0.50),
        ("trend_strength", 'features["trend_strength"]', -0.40, 0.40),
    ],
    "uncertainty_gated": [
        ("u_centered", '(features["u"] - 0.5)', -0.60, 0.60),
        ("crisis_confidence", 'features["crisis"] * (1.0 - features["uncertainty"])', -0.60, 0.60),
        ("up_confidence", 'features["trend_up"] * (1.0 - features["uncertainty"])', -0.50, 0.50),
        ("down_confidence", 'features["trend_down"] * (1.0 - features["uncertainty"])', -0.50, 0.50),
        ("uncertainty", 'features["uncertainty"]', -0.60, 0.60),
    ],
    "trend_emphasis": [
        ("u_centered", '(features["u"] - 0.5)', -0.50, 0.50),
        ("trend_up", 'features["trend_up"]', -0.60, 0.60),
        ("trend_down", 'features["trend_down"]', -0.60, 0.60),
        ("trend_strength", 'features["trend_strength"]', -0.60, 0.60),
        ("uncertainty", 'features["uncertainty"]', -0.30, 0.30),
    ],
    "asymmetric_interaction": [
        ("u_centered", '(features["u"] - 0.5)', -0.60, 0.60),
        ("crisis", 'features["crisis"]', -0.60, 0.60),
        ("trend_up", 'features["trend_up"]', -0.40, 0.40),
        ("trend_down", 'features["trend_down"]', -0.40, 0.40),
        ("uncertainty", 'features["uncertainty"]', -0.60, 0.60),
        ("u_uncertainty", '(features["u"] - 0.5) * features["uncertainty"]', -0.60, 0.60),
    ],
    "volatility_regime": [
        ("u_centered", '(features["u"] - 0.5)', -0.60, 0.60),
        ("volatility", 'features["volatility_20d"]', -0.50, 0.50),
        ("crisis", 'features["crisis"]', -0.60, 0.60),
        ("trend_strength", 'features["trend_strength"]', -0.40, 0.40),
        ("uncertainty", 'features["uncertainty"]', -0.60, 0.60),
    ],
    "piecewise_regime": [
        ("u_centered", '(features["u"] - 0.5)', -0.60, 0.60),
        ("crisis_indicator", '(features["crisis"] > 0.5).astype(float)', -0.50, 0.50),
        ("up_indicator", '(features["trend_up"] > features["trend_down"]).astype(float)', -0.50, 0.50),
        ("uncertainty", 'features["uncertainty"]', -0.60, 0.60),
        ("trend_strength", 'features["trend_strength"]', -0.40, 0.40),
    ],
}

TEMPLATE = Template('''"""Generated RUTL AlphaEvolve candidate."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

POLICY_FAMILY = "$family"
FORMULA_TERMS = $terms
FIT_SEED = $fit_seed
N_TRIALS = $n_trials


class ThresholdPolicy:
    def __init__(self) -> None:
        self.params = None
        self.bounds = ($q_low, $q_high)

    @staticmethod
    def _score(actions, returns):
        changes = np.abs(np.diff(np.r_[0.0, actions]))
        gross = actions * returns - 0.001 * changes
        equity = np.cumprod(1.0 + gross)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (peaks - equity) / np.maximum(peaks, 1e-12)
        mean = float(np.mean(gross) * 252.0)
        vol = float(np.std(gross) * np.sqrt(252.0))
        return mean, vol, float(np.max(drawdowns))

    def _q(self, features, params):
        q = np.full(len(features["u"]), params[0], dtype=float)
        for i, (_, expression, _, _) in enumerate(FORMULA_TERMS):
            q = q + params[i + 1] * eval(expression, {"__builtins__": {}}, {"features": features, "np": np})
        return np.clip(q, *self.bounds)

    def fit(self, train_data, valid_data, config):
        rng = np.random.default_rng(FIT_SEED)
        lower = np.array([$q_low] + [term[2] for term in FORMULA_TERMS])
        upper = np.array([$q_high] + [term[3] for term in FORMULA_TERMS])
        best_score = -np.inf
        best = np.r_[0.50, np.zeros(len(FORMULA_TERMS))]
        for _ in range(N_TRIALS):
            params = rng.uniform(lower, upper)
            q_train = self._q(train_data, params)
            q_valid = self._q(valid_data, params)
            train_actions = (train_data["u"] >= q_train).astype(float)
            valid_actions = (valid_data["u"] >= q_valid).astype(float)
            train_mean, train_vol, train_dd = self._score(train_actions, train_data["r_next"])
            valid_mean, valid_vol, valid_dd = self._score(valid_actions, valid_data["r_next"])
            score = (
                $train_weight * train_mean
                + $valid_weight * valid_mean
                - $vol_weight * valid_vol
                - $dd_weight * valid_dd
            )
            if score > best_score:
                best_score = score
                best = params
        self.params = best

    def predict_threshold(self, features):
        if self.params is None:
            raise RuntimeError("fit must be called before predict_threshold")
        return self._q(features, self.params)

    def state_dict(self):
        return {
            "family": POLICY_FAMILY,
            "params": None if self.params is None else self.params.tolist(),
        }
''')


def call(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, cwd=str(ROOT))


def sample_parent(run_dir: Path) -> tuple[str | None, dict]:
    output = call(["python3", str(SKILL), "sample", "--run", str(run_dir), "--count", "3", "--mode", "balanced"])
    data = json.loads(output)
    candidates = []
    if data.get("parent") is not None:
        candidates.append(data["parent"])
    candidates.extend(data.get("inspirations", []))
    valid = [c for c in candidates if c.get("valid")]
    if not valid:
        return None, data
    best = max(valid, key=lambda c: float(c.get("scores", {}).get("nu_lb", -1e9)))
    return best["id"], data


def choose_family(parent: dict | None, rng: np.random.Generator, role: str) -> str:
    if role == "exploit" and parent is not None:
        family = parent.get("behavior", {}).get("policy_family")
        if family in FAMILIES:
            return family
    return str(rng.choice(list(FAMILIES)))


def make_candidate(run_dir: Path, gen: int, slot: int, family: str, rng: np.random.Generator, parent_id: str | None) -> Path:
    terms = FAMILIES[family]
    source = TEMPLATE.substitute(
        family=family,
        terms=repr(terms),
        fit_seed=int(rng.integers(1, 2**31 - 1)),
        n_trials=int(rng.choice([160, 240, 320, 480])),
        q_low=0.05,
        q_high=0.95,
        train_weight=float(rng.choice([0.15, 0.30, 0.45])),
        valid_weight=float(rng.choice([0.55, 0.70, 0.85])),
        vol_weight=float(rng.choice([0.02, 0.05, 0.10])),
        dd_weight=float(rng.choice([0.05, 0.10, 0.20])),
    )
    candidate_dir = run_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    path = candidate_dir / f"g{gen:04d}-c{slot:02d}.py"
    path.write_text(source, encoding="utf-8")
    return path


def archive(run_dir: Path, candidate: Path, result: dict, generation: int, parent_id: str | None, feedback: str) -> None:
    scores = result["scores"]
    behavior = result.get("behavior", {})
    cmd = [
        "python3",
        str(SKILL),
        "add",
        "--run",
        str(run_dir),
        "--candidate",
        str(candidate),
        "--scores",
        json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in scores.items()}),
        "--behavior",
        json.dumps(behavior),
        "--generation",
        str(generation),
        "--feedback",
        feedback,
    ]
    if parent_id:
        cmd += ["--parent", parent_id]
    if not result.get("valid", False):
        cmd.append("--invalid")
    call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=str(ROOT / "alpha_evolve" / "run_v1"))
    parser.add_argument("--max-generations", type=int, default=60)
    parser.add_argument("--no-improvement", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()

    run_dir = Path(args.run)
    rng = np.random.default_rng(args.seed)
    best_score = -1e9
    stagnant = 0
    for gen in range(1, args.max_generations + 1):
        parent_id, sample_data = sample_parent(run_dir)
        parent = sample_data.get("parent") if sample_data else None
        slots = [("exploit", 1), ("explore", 2), ("repair", 3)]
        generation_best = best_score
        for role, slot in slots:
            family = choose_family(parent, rng, role)
            candidate = make_candidate(run_dir, gen, slot, family, rng, parent_id)
            started = time.time()
            result = evaluate_candidate(candidate, BUNDLE, PROTOCOL)
            feedback = f"role={role}; family={family}; runtime={time.time()-started:.2f}s"
            archive(run_dir, candidate, result, gen, parent_id, feedback)
            score = float(result["scores"].get("nu_lb", -1e9))
            generation_best = max(generation_best, score)
            print(json.dumps({
                "generation": gen,
                "slot": slot,
                "role": role,
                "family": family,
                "valid": result["valid"],
                "nu_lb": score,
                "delta_vs_best_fixed": result["scores"].get("delta_vs_best_fixed"),
                "correctness": result["scores"].get("correctness"),
            }, ensure_ascii=False), flush=True)
        if generation_best > best_score + 1e-4:
            best_score = generation_best
            stagnant = 0
        else:
            stagnant += 1
            print(json.dumps({"generation": gen, "stagnant": stagnant}), flush=True)
        if stagnant >= args.no_improvement:
            print(json.dumps({"stopping": "no_improvement", "generation": gen, "best_score": best_score}), flush=True)
            break

    print(call(["python3", str(SKILL), "status", "--run", str(run_dir)]))


if __name__ == "__main__":
    main()
