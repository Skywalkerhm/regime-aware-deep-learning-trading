from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from src.rutl.backtest import evaluate_actions
from src.rutl.metrics import block_bootstrap_lower_bound


class MetricsTest(unittest.TestCase):
    def test_all_long_equals_return_minus_entry_cost(self) -> None:
        returns = np.array([0.01, -0.005, 0.02, 0.0])
        result = evaluate_actions(np.ones(4), returns, transaction_cost=0.001, drawdown_penalty=0.0)
        expected = np.prod(1.0 + np.r_[returns[0] - 0.001, returns[1:]]) - 1.0
        self.assertAlmostEqual(result.total_return, expected, places=10)

    def test_bootstrap_is_deterministic(self) -> None:
        values = np.linspace(-0.01, 0.02, 200)
        a = block_bootstrap_lower_bound(values, replications=100, seed=7)
        b = block_bootstrap_lower_bound(values, replications=100, seed=7)
        self.assertEqual(a, b)

    def test_action_shape_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_actions(np.ones(3), np.ones(2))


if __name__ == "__main__":
    unittest.main()
