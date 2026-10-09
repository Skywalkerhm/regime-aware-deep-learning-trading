from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class AppliedResultsTest(unittest.TestCase):
    def test_bundle_dates_are_complete(self) -> None:
        bundle = pd.read_csv(ROOT / "artifacts" / "bundles" / "predictions_v1.csv", parse_dates=["date"], low_memory=False)
        real = bundle[bundle["asset"].isin(["BTC-USD", "TSLA"])]
        self.assertFalse(real["date"].isna().any())

    def test_group_results_match_manuscript(self) -> None:
        pooled = pd.read_csv(ROOT / "artifacts" / "results" / "applied_fixed_vs_rule_v1.csv")
        lookup = pooled.set_index(["scope", "policy"])["mean_nu_lb"].to_dict()
        self.assertAlmostEqual(lookup[("search", "fixed_validation_optimized")], -0.095340, places=5)
        self.assertAlmostEqual(lookup[("search", "current_rule")], -0.383652, places=5)
        self.assertAlmostEqual(lookup[("validation", "fixed_validation_optimized")], 0.014386, places=5)
        self.assertAlmostEqual(lookup[("validation", "current_rule")], -0.352445, places=5)
        self.assertAlmostEqual(lookup[("external", "fixed_validation_optimized")], -0.093521, places=5)
        self.assertAlmostEqual(lookup[("external", "current_rule")], -0.612740, places=5)

    def test_external_interval_excludes_zero_and_is_negative(self) -> None:
        paired = pd.read_csv(ROOT / "artifacts" / "results" / "applied_paired_bootstrap_v1.csv")
        external = paired[paired["scope"] == "external"].iloc[0]
        self.assertLess(external["ci_high"], 0.0)

    def test_group_bootstrap_supports_fixed(self) -> None:
        summary = pd.read_csv(ROOT / "artifacts" / "results" / "applied_group_bootstrap_v1.csv").iloc[0]
        self.assertAlmostEqual(summary["mean_difference"], -0.519219, places=5)
        self.assertLess(summary["ci_high"], 0.0)

    def test_placebo_is_not_decisive(self) -> None:
        placebo = pd.read_csv(ROOT / "artifacts" / "results" / "applied_placebo_rule_labels_v1.csv")
        self.assertAlmostEqual(placebo["percentile"].mean(), 0.535892, places=5)
        self.assertLess((placebo["percentile"] > 0.95).sum(), 2)

    def test_cost_sensitivity_is_ordered(self) -> None:
        costs = pd.read_csv(ROOT / "artifacts" / "results" / "applied_cost_sensitivity_v1.csv")
        zero = costs[costs["cost_bps"] == 0].set_index("policy")["mean_nu_lb"]
        hundred = costs[costs["cost_bps"] == 100].set_index("policy")["mean_nu_lb"]
        self.assertGreater(zero["fixed_validation_selected"], zero["current_rule"])
        self.assertGreater(hundred["fixed_validation_selected"], hundred["current_rule"])

    def test_drrl_strong_control(self) -> None:
        import json
        data = json.loads((ROOT / "artifacts" / "results" / "drrl_best_config_strong_20seed.json").read_text())
        rows = [x for x in data if x["scenario"] == "decision_relevant_state"]
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(row["activated"] for row in rows))
        self.assertGreater(sum(row["drrl_delta"] for row in rows) / len(rows), 0.5)

    def test_drrl_uninformative_collapse(self) -> None:
        import json
        data = json.loads((ROOT / "artifacts" / "results" / "drrl_best_config_strong_20seed.json").read_text())
        rows = [x for x in data if x["scenario"] == "uninformative_regime"]
        self.assertTrue(all(not row["activated"] for row in rows))
        self.assertTrue(all(abs(row["drrl_delta"]) < 1e-12 for row in rows))

    def test_validation_tuning_does_not_rescue_rule(self) -> None:
        tuned = pd.read_csv(ROOT / "artifacts" / "results" / "applied_tuned_regime_v1.csv").set_index("policy")
        fixed = tuned.loc["fixed_validation_selected", "mean_nu_lb"]
        self.assertGreater(fixed, tuned.loc["validation_tuned_global", "mean_nu_lb"])
        self.assertGreater(fixed, tuned.loc["validation_tuned_per_model", "mean_nu_lb"])

    def test_rule_wins_one_of_fourteen_groups(self) -> None:
        groups = pd.read_csv(ROOT / "artifacts" / "results" / "applied_external_by_fold_model_v1.csv")
        self.assertEqual(len(groups), 14)
        self.assertEqual(int((groups["difference"] > 0).sum()), 1)


if __name__ == "__main__":
    unittest.main()
