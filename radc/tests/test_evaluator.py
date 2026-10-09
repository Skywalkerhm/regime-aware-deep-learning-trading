from __future__ import annotations

import sys
import unittest
from pathlib import Path

from src.rutl.evaluator import evaluate_candidate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class EvaluatorTest(unittest.TestCase):
    def test_known_good_is_valid(self) -> None:
        result = evaluate_candidate(
            ROOT / "alpha_evolve" / "candidates" / "g0000_known_good.py",
            ROOT / "artifacts" / "bundles" / "predictions_v1.csv",
            ROOT / "configs" / "evaluation_protocol.json",
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["scores"]["correctness"], 1.0)

    def test_known_bad_is_rejected(self) -> None:
        result = evaluate_candidate(
            ROOT / "alpha_evolve" / "candidates" / "g0000_known_bad.py",
            ROOT / "artifacts" / "bundles" / "predictions_v1.csv",
            ROOT / "configs" / "evaluation_protocol.json",
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["scores"]["correctness"], 0.0)


if __name__ == "__main__":
    unittest.main()
