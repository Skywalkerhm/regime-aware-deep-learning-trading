# Supporting Information (PLOS ONE)

List the items exactly as they should appear after the main text.

- **S1 Text. Proofs of Propositions 1-5 and the finite-sample bound.** Contains the
  full proofs, including Proposition 4's strictness caveat and the isotonic
  counterexample (`docs/theory_formal.md`).
- **S1 Data. Frozen prediction bundle and result tables.** `predictions_v1.csv`,
  `artifacts/results/*.csv`, and `artifacts/bundles/manifest_applied_v1.json`.
- **S1 Code. Analysis code.** `src/rutl/`, `experiments/`, `tests/`.
- **S1 Fig. Decision-regret mechanism.** Panel A: conditional action value `Delta(p,z)`
  for a decision-relevant versus a state-invariant DGP. Panel B: decision-regret curves
  with the value-of-information gap. File:
  `artifacts/reports/figures/decision_regret_mechanism.png`.
- **S2 Fig. Mechanism confirmation.** Regret reduction versus realized gain
  (r = +0.94) and regret reduction versus ECE (r = +0.10). File:
  `artifacts/reports/figures/decision_regret_confirmation.png`.
- **S3 Fig. Oracle upper bound.** Recovery of the oracle lower-bound gain and the
  oracle ceiling versus captured gain. File:
  `artifacts/reports/figures/oracle_upper_bound.png`.

## Figure captions (main text)

**Fig 1.** Decision-regret mechanism. (A) Conditional action value `Delta(p,z)` for the
decision-relevant DGP (two states cross zero at different probabilities) and for a
state-invariant DGP (curves coincide). (B) Decision regret of a single fixed threshold;
the gap between the fixed-threshold minimum and the state-aware minimum is the value of
information.

**Fig 2.** Mechanism confirmation across 160 synthetic rows. (A) Oracle regret
reduction versus realized external gain (r = +0.94). (B) Regret reduction (blue) versus
ECE (red) by scenario; ECE is flat and does not track the gain.

**Fig 3.** Oracle upper bound. (A) RADC realized gain versus the oracle lower-bound gain
with the y = x line (r = +0.99). (B) Oracle ceiling versus captured gain by scenario.
