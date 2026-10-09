"""Known-good correctness candidate: always a valid constant threshold."""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np

class ThresholdPolicy:
    def fit(self, train_data: Mapping[str, np.ndarray], valid_data: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> None:
        self.threshold = 0.70
    def predict_threshold(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.full(len(features["u"]), self.threshold, dtype=float)
    def state_dict(self) -> dict[str, Any]:
        return {"threshold": self.threshold}
