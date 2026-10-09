# Cover Letter — PLOS ONE

9 October 2026

The Editors
PLOS ONE

**Re: Submission of a Research Article — "Regime-aware decision calibration: when
should state information change a trading threshold?"**

**Article type:** Research Article.

Dear Editors,

I submit for your consideration the Research Article **"Regime-aware decision
calibration: when should state information change a trading threshold?"**

## Summary of the study

Conditioning a trading threshold on a detected market "regime" is a widely used,
model-agnostic intervention, but it frequently fails out of sample and its mechanism is
poorly understood. This paper argues that the objects usually invoked to justify it —
probability calibration and regime-classification accuracy — are not the decision
mechanism. The relevant object is **decision calibration**: a state partition improves
decisions only if it changes the optimal action on a positive-probability region whose
expected utility exceeds the estimation and risk costs.

I formalize this as a decision-regret and value-of-information problem, define
**Regime-Aware Decision Calibration (RADC)**, and prove five results, including two
identification/negative statements: probability calibration cannot certify decision
relevance, and a strictly monotone recalibration is decision-inert for re-optimized
thresholds. RADC is a two-gate estimator: it selects a resampling-stable context
partition, then activates state-dependent thresholds only when a lower confidence bound
on the value of state information exceeds a conformal risk buffer; otherwise it
collapses to the best fixed threshold.

## Principal results

- The decision-regret reduction tracks the realized out-of-sample gain (r = +0.945),
  whereas expected calibration error does not (r = +0.098).
- Probability calibration is decision-inert: temperature and Platt scaling change
  external utility by exactly 0.0000 (on real data, ECE falls from 0.074 to 0.017 with
  no change in utility), while decision calibration adds +0.2881 in a 2x2 factorization
  with zero interaction.
- On controlled regime data-generating processes the estimator recovers essentially the
  full oracle gain (recovery 1.01 on a strong positive control), activates in 0% of
  decision-irrelevant and weak-detector controls, loses its gain under state-shuffle
  placebos, and abstains on 100% of cases with no positive oracle gain.
- On a frozen Bitcoin/Tesla bundle the oracle upper bound is negative, and the method
  correctly abstains rather than degrading the fixed threshold.

## Relation to previous work

The paper builds on regime-switching models, probability calibration, value-of-
information theory, and conformal risk control, and it contributes the decision-
theoretic object that connects regime learning to threshold decisions. It does not
claim a profitable trading strategy; the contribution is a testable criterion for
*whether to be regime-aware at all*, together with controlled evidence for the
mechanism.

## Why PLOS ONE

The study is a methods-and-mechanism contribution whose claims are bounded by the
evidence. Every result reproduces bit-identically from a frozen bundle, and the
submission includes a one-command reproduction package, a SHA-256 manifest, and an
automated test suite. I have reported the negative real-asset outcome as a boundary
condition rather than concealing it.

## Suggested Academic Editors

[[Optional: name 1-3 PLOS ONE Academic Editors whose expertise fits decision theory,
machine learning in finance, or calibration; see the PLOS ONE Editorial Board list.]]

## Opposed reviewers

[[Optional: list any individuals who should not review this manuscript, with a brief
reason.]]

## Declarations

The manuscript is original, has not been published previously, and is not under
consideration elsewhere. The single author has approved the submission and declares no
competing interests.

Sincerely,

Min Huang
The Chinese University of Hong Kong, Shenzhen, Guangdong, China
minhuang1@link.cuhk.edu.cn
ORCID: 0009-0007-4991-3221
