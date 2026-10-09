from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.activation_rates import wilson


class ActivationRatesTest(unittest.TestCase):
    def test_wilson_bounds(self) -> None:
        lo, hi = wilson(0, 69)
        self.assertLess(lo, 1e-6)
        self.assertLess(hi, 0.06)  # 0/69 upper bound ~5.3%
        lo2, hi2 = wilson(69, 69)
        self.assertGreater(lo2, 0.9)
        self.assertEqual(hi2, 1.0)
        # symmetric-ish sanity
        lo3, hi3 = wilson(5, 10)
        self.assertLess(lo3, 0.5)
        self.assertGreater(hi3, 0.5)

    def test_activation_table_consistency(self) -> None:
        table = pd.read_csv(ROOT / "artifacts" / "results" / "activation_rates.csv")
        neg = table[table["label"] == "negative_controls_pooled"]
        self.assertEqual(len(neg), 1)
        self.assertEqual(int(neg["activated"].iloc[0]), 0)
        self.assertLess(float(neg["wilson_hi"].iloc[0]), 0.06)
        near = table[table["label"] == "abstain_when_near_zero"]
        self.assertEqual(int(near["activated"].iloc[0]), int(near["runs"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
