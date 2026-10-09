#!/usr/bin/env python3
"""Create SHA-256 manifests for the current RUTL and applied-paper artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMON = [
    "artifacts/bundles/synthetic_v1.csv",
    "artifacts/bundles/real_v1.csv",
    "artifacts/bundles/predictions_v1.csv",
    "artifacts/results/champion_v1_external.json",
    "artifacts/results/champion_v1_external_by_model.json",
    "artifacts/candidates/champion_v1.py",
    "alpha_evolve/run_v1/status_final.json",
    "alpha_evolve/run_v1/frontier.json",
    "src/rutl/policy.py",
]
APPLIED = [
    "literature_review/paper_matrix.csv",
    "literature_review/synthesis.md",
    "literature_review/direction_report.md",
    "alpha_evolve/direction_run_v1/status_final.json",
    "direction_evolve2/positive_method_plan.md",
    "docs/theory_notes.md",
    "docs/theory_formal.md",
    "docs/mechanism_confirmation_plan.md",
    "alpha_evolve/method_direction_run_v1/status_final.json",
    "alpha_evolve/direction_run_v1/frontier.json",
    "applied_paper/manuscript_v1.md",
    "applied_paper/evidence_map.md",
    "applied_paper/revision_checklist.md",
    "artifacts/results/applied_fixed_vs_rule_v1.csv",
    "artifacts/results/applied_external_by_fold_model_v1.csv",
    "artifacts/results/applied_paired_bootstrap_v1.csv",
    "artifacts/results/applied_external_groups_v1.csv",
    "artifacts/results/applied_group_bootstrap_v1.csv",
    "artifacts/results/applied_static_baselines_v1.csv",
    "artifacts/results/applied_cost_sensitivity_v1.csv",
    "artifacts/results/applied_placebo_rule_labels_v1.csv",
    "artifacts/results/applied_tuned_regime_v1.csv",
    "artifacts/results/applied_tuned_regime_per_model_v1.csv",
    "artifacts/reports/applied_verification_report.md",
    "artifacts/reports/drrl_v0_report.md",
    "artifacts/reports/drrl_v0_final_test.md",
    "artifacts/reports/drrl_v2_report.md",
    "artifacts/results/drrl_v0_test.json",
    "artifacts/results/drrl_v0_sweep.csv",
    "artifacts/results/drrl_controls_20seed.csv",
    "artifacts/results/drrl_best_config_strong_20seed.csv",
    "src/rutl/decision_regret.py",
    "experiments/decision_regret_experiment.py",
    "experiments/decision_regret_baselines.py",
    "tests/test_decision_regret.py",
    "artifacts/results/decision_regret_synthetic.csv",
    "artifacts/results/decision_regret_real.csv",
    "artifacts/results/decision_regret_curves.csv",
    "artifacts/results/decision_regret_summary.json",
    "artifacts/results/decision_regret_baselines.csv",
    "artifacts/reports/decision_regret_report.md",
    "experiments/calibration_decision_2x2.py",
    "experiments/plot_decision_regret.py",
    "tests/test_calibration_invariance.py",
    "direction_evolve2/experiment_b_drrl_config.json",
    "artifacts/results/calibration_decision_2x2.csv",
    "artifacts/results/calibration_decision_2x2_summary.json",
    "artifacts/reports/calibration_decision_2x2_report.md",
    "artifacts/reports/figures/decision_regret_mechanism.png",
    "artifacts/reports/figures/decision_regret_confirmation.png",
    "experiments/oracle_upper_bound.py",
    "experiments/stability_placebo.py",
    "experiments/plot_oracle_upper_bound.py",
    "tests/test_oracle_stability.py",
    "artifacts/results/oracle_upper_bound.csv",
    "artifacts/results/oracle_upper_bound_summary.json",
    "artifacts/results/stability_placebo.csv",
    "artifacts/results/stability_placebo_summary.json",
    "artifacts/results/split_stability.csv",
    "artifacts/reports/oracle_upper_bound_report.md",
    "artifacts/reports/stability_placebo_report.md",
    "artifacts/reports/figures/oracle_upper_bound.png",
    "method_paper/manuscript_v1.md",
    "method_paper/evidence_map.md",
    "method_paper/revision_checklist.md",
    "method_paper/README.md",
    "method_paper/plos_one/manuscript_plos_one.md",
    "method_paper/plos_one/fit_assessment.md",
    "method_paper/plos_one/self_review.md",
    "experiments/calibration_method_sensitivity.py",
    "artifacts/results/calibration_method_sensitivity.csv",
    "experiments/bin_count_sensitivity.py",
    "artifacts/results/bin_count_sensitivity.csv",
    "artifacts/results/bin_count_sensitivity_summary.csv",
    "experiments/activation_rates.py",
    "artifacts/results/activation_rates.csv",
    "artifacts/results/activation_rates.md",
    "method_paper/references_vancouver.md",
    "tests/test_activation_rates.py",
    "method_paper/plos_one/submission/title_page.md",
    "method_paper/plos_one/submission/cover_letter.md",
    "method_paper/plos_one/submission/statements.md",
    "method_paper/plos_one/submission/supporting_information.md",
    "method_paper/plos_one/submission/submission_checklist.md",
    "artifacts/reports/data_verification_report.md",
]


def entry(relative: str) -> dict[str, object]:
    path = ROOT / relative
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def write_manifest(name: str, files: list[str]) -> None:
    payload = {
        "protocol_version": "2026-09-28-rutl-v1",
        "files": {relative: entry(relative) for relative in files},
    }
    (ROOT / "artifacts" / "bundles" / name).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    write_manifest("manifest_v1.json", COMMON)
    write_manifest("manifest_applied_v1.json", COMMON + APPLIED)


if __name__ == "__main__":
    main()
