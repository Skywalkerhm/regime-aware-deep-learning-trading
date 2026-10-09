from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from src.rutl.data import feature_matrix

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


class CausalityTest(unittest.TestCase):
    def test_candidate_features_exclude_labels_and_returns(self) -> None:
        frame = pd.read_csv(ROOT / "artifacts" / "bundles" / "predictions_v1.csv", low_memory=False).head(10)
        features = feature_matrix(frame)
        self.assertNotIn("y", features)
        self.assertNotIn("r_next", features)
        self.assertNotIn("target", features)

    def test_external_rows_are_not_used_for_fit(self) -> None:
        frame = pd.read_csv(ROOT / "artifacts" / "bundles" / "predictions_v1.csv", low_memory=False)
        search = frame[frame["split"] == "search"]
        external = frame[frame["split"] == "external"]
        self.assertFalse(set(search.index) & set(external.index))


if __name__ == "__main__":
    unittest.main()
