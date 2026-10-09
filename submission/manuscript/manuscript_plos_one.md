# Regime-aware decision calibration: when should state information change a trading threshold?

**Short title:** Regime-aware decision calibration

**Authors:** Min Huang^1\*

**Affiliations:**
1. The Chinese University of Hong Kong, Shenzhen, Guangdong, China

**\* Corresponding author:** minhuang1@link.cuhk.edu.cn, ORCID 0009-0007-4991-3221

**Draft:** PLOS ONE submission draft v2
**Date:** 2026-10-09

## Abstract

Conditioning a trading threshold on a detected market regime is a widely used,
model-agnostic intervention, yet it frequently fails out of sample and its mechanism
is poorly specified. We argue that the objects usually invoked to justify it —
probability calibration and statistical regime accuracy — are not the decision
mechanism. The relevant object is *decision calibration*: a state partition improves
decisions only if it changes the optimal action on a positive-probability region whose
expected utility exceeds the estimation and risk costs. We formalize this as a
decision-regret and value-of-information problem and define Regime-Aware Decision
Calibration (RADC), a two-gate estimator that (i) selects a resampling-stable context
partition and (ii) activates state-dependent thresholds only when a lower confidence
bound on the value of state information exceeds a conformal risk buffer, otherwise
collapsing to the best fixed threshold. We prove five results: the utility gain equals
the decision-regret reduction minus cost; probability calibration cannot certify
decision relevance; a strictly monotone recalibration is decision-inert for
re-optimized thresholds; partition stability does not imply decision relevance; and
abstention is Bayes-optimal when the value of information is non-positive. On controlled regime data-generating processes the decision-regret reduction tracks
the realized gain (r = +0.945) while expected calibration error does not (r = +0.098);
the estimator recovers the oracle decision gain (recovery 1.01 on a strong positive
control), activates in 0% of decision-irrelevant and weak-detector controls, and loses
its gain under state-shuffle placebos. In a 2x2 factorization, decision calibration adds utility identically before
and after probability calibration (interaction 0.0000), whereas probability
calibration leaves utility unchanged. On a frozen Bitcoin/Tesla bundle the oracle
upper bound is negative and the method correctly abstains. The contribution is a
principled, testable way to decide whether to be regime-aware at all.

**Keywords:** regime switching; decision calibration; value of information; threshold
optimization; conformal risk control; walk-forward evaluation

## Introduction

Machine-learning trading pipelines typically separate forecasting from decision making.
A model estimates the probability that the next return is positive, and a decision rule
maps that probability to a position. When a fixed cutoff performs poorly, a natural
remedy is to make the cutoff state-dependent: lower the entry threshold in trending
markets, raise it in crises, and use an intermediate value otherwise. The intervention
is attractive because it is post-hoc, model-agnostic, and requires no retraining.

It nevertheless fails more often than its popularity suggests. In a reproducible
reassessment on Bitcoin and Tesla we found that a rule-based regime threshold was
decisively worse than a single validation-selected fixed threshold, in validation and
on untouched external folds (external mean group net-utility lower bound -0.613 versus
-0.094; the rule won 1 of 14 asset-model-fold groups). A common explanation invokes
probability calibration, but a regime label can be statistically predictable and still
carry no decision value; and, as we prove, monotone recalibration cannot change the
decision value of a threshold rule at all.

This paper isolates the missing object. A state partition improves decisions if and
only if it changes the *optimal action* on a region with positive expected utility.
The correct quantity is therefore a decision-regret (or value-of-information) object,
not a calibration statistic. We then design an estimator that must *earn* the right to
use regimes.

**Related work.** Regime-switching models formalize time-varying return and volatility
dynamics [1,2]. Recent work applies regime awareness at the architecture or policy
level: sectoral portfolio gating [3], regime-aware architecture search [4],
regime-gated attention [5], and regime-aware reinforcement learning [6]. Threshold
optimization has been studied directly, e.g. in high-frequency pairs trading [7] and
profit-optimized Bitcoin thresholds [8]. Neural forecasts are often miscalibrated
[9,10], motivating partial-calibration and decision-robustness methods [11], inverse
conformal risk control [12], inference-time decision calibration [13], and evidence
that forecast skill is not decision skill [14]; regime-aware conformal calibration has
been applied to economic forecasting [15]. Value of information is classical [16], and
conformal prediction provides distribution-free risk control [17]. Deep learning is
widely used for financial prediction [18]. Financial backtests are vulnerable to
selection and overfitting [19,20], motivating leakage-safe, search-aware evaluation
[21,22] and cost-aware walk-forward studies [23]. Related feature-selection work
includes distributional Markov-chain correction [24]. We contribute the decision-
theoretic object that connects regime learning to threshold decisions, and prove when
calibration-based justifications are insufficient.

**Contributions.**

1. A decision-calibration formulation of regime thresholds: conditional action value
   `Delta(p,z)`, decision regret `R_dec`, and value of information `VoI`, with the
   identity `V_state - V_fixed = R_dec(theta_fixed) - R_dec(theta_state) - cost`.
2. A negative identification result: probability calibration cannot certify decision
   relevance, and monotone recalibration is decision-inert for re-optimized thresholds.
3. A state-relevance separation condition, and the result that partition stability does
   not imply decision relevance.
4. A two-gate estimator (RADC) with a value-of-information activation gate and a
   conformal risk buffer; collapse to the fixed threshold is Bayes-optimal when
   `VoI <= 0`.
5. A controlled validation with positive and negative controls, a 2x2 calibration
   factorization, an oracle upper bound, and state-shuffle placebos.

## Materials and Methods

### Data and evaluation protocol

We use a frozen probability bundle. Bitcoin models are LSTM, Transformer, and XGBoost;
Tesla uses an LSTM. Base-model parameters are frozen before the threshold comparison;
the method only selects a context partition and thresholds. The evaluation uses a
long/flat action, a 10 basis-point transaction cost, and a drawdown penalty of 0.25;
the primary metric is a group-level net-utility lower bound (`NU_LB`) computed by a
moving-block bootstrap (block 10, 2000 replications, 5% quantile). The evaluation unit
is `asset x model x fold`, and groups are averaged equally. Search, validation, and
external splits are strictly separated; the fixed threshold is selected on validation
only. In addition to the real bundle we use a set of synthetic regime data-generating
processes (DGPs) with a held-out tail. No external assets are used. The protocol is
`configs/evaluation_protocol.json`, the splits are `configs/data_splits.json`, and the
frozen bundle is `artifacts/bundles/predictions_v1.csv`.

### Problem setup

Let `p_t` be a base-model forecast probability, `x_t` an observable context vector,
`z_t = g(x_t)` a latent decision state, `a_t in {0,1}` a long/flat action, and
`r_{t+1}` the next-period return. Net per-period utility after costs is
`U_t(a_t, r_{t+1})`. A fixed-threshold policy is `pi_f = 1[p_t >= theta]` and a
state-dependent policy is `pi_s = 1[p_t >= theta(z_t)]`.

### Decision regret and value of information

Define the conditional action value and the Bayes action:

```text
Delta(p, z) = E[U(1, R) | P=p, Z=z] - E[U(0, R) | P=p, Z=z]
a*(p, z)    = 1[Delta(p, z) >= 0].
```

Decision regret of a threshold rule `theta` is the expected utility lost by taking the
wrong action:

```text
R_dec(theta) = E[ max(0, -Delta(P,Z)) * 1[P >= theta(Z)]
                + max(0,  Delta(P,Z)) * 1[P <  theta(Z)] ].
```

The value of regime information is

```text
VoI(theta) = R_dec(theta_fixed) - R_dec(theta_state) - C(theta_state),
```

where `C` collects expected turnover and drawdown cost. Deployment uses

```text
use theta_state  iff  LCB(VoI) > R_alpha,
```

with `R_alpha` a split-conformal risk buffer; otherwise the method collapses to
`theta_fixed`. `Delta` is estimated in probability bins on the search+validation data.

### The two-gate estimator (RADC)

1. **Identification gate.** A one-level decision-relevant split `g(x)` is proposed on
   the search split and retained only if it survives bootstrap stability selection
   (feature and cutoff selection frequency above a threshold), excluding unstable,
   noise-driven partitions.
2. **Decision gate.** On the validation split the incremental utility of the split
   policy over the fixed threshold is estimated per group; activation requires
   `LCB(VoI) > R_alpha` and a minimum fraction of positive groups.

Both gates must pass; the decision gate is the operative decision-calibration
criterion.

**Algorithm 1 (RADC).**
```text
Input: search S, validation V, external E; score p; context x; threshold grid Theta
1  estimate Delta(p,z) in probability bins from S
2  fit a one-level decision-relevant split g(x) on S by minimising decision regret
3  resample S; keep g only if feature and cutoff are stable (frequency > tau)
   if no stable g: return fixed policy pi_f
4  select the fixed threshold theta_f on V by the lower-bound net utility
5  per group in V: d_g = utility(g) - utility(theta_f)
6  LCB = lower confidence bound of {d_g}; R_alpha = conformal risk buffer on V
7  if LCB > R_alpha and positive-group fraction >= rho:
       output state policy pi_s = 1[p >= theta(z)]
   else:
       output fixed policy pi_f = 1[p >= theta_f]      # abstention
```

### Theoretical analysis

**Proposition 1 (VoI is decision regret, not calibration).** For any rule `theta`,
`V(theta) = E[max(Delta,0)] - R_dec(theta) - C(theta)`, hence
`V_state - V_fixed = R_dec(theta_fixed) - R_dec(theta_state) - C(theta_state)`.

**Proposition 2 (probability calibration does not certify decision relevance).** ECE
and reliability diagrams constrain only the marginal `P(Y=1|P=p)`. `Delta(p,z)` also
depends on the state-conditional payoff asymmetry, which is not identified by the
calibrated marginal. Two DGPs can share identical ECE and Brier scores while having
`VoI = 0` and `VoI > 0`, and the gap is unbounded.

**Proposition 3 (state-relevance separation).** A refinement strictly improves the
Bayes value only if there is a positive-probability set `S` on which the optimal
actions differ across states and `E[|Delta| 1_S] > C(theta_state)`. Statistical
separability of the state-conditional forecast distributions is neither necessary nor
sufficient.

**Proposition 4 (strictly monotone recalibration is decision-inert).** For any
strictly increasing `phi`, `1[phi(p) >= phi(theta)] <=> 1[p >= theta]`, so the
achievable action set is invariant: temperature and Platt scaling cannot change the
value of a re-optimized threshold. Isotonic regression is only *non-strictly*
monotone; its flat regions coarsen the cutpoints, and a sensitivity check shows it
*can* change (here, lower) the utility (see Results and S1 Text).

**Proposition 5 (stability does not imply decision relevance).** Stability concerns the
identification of `g`; `VoI` is a functional of `Delta(p,z)`. Stable partitions with
`VoI = 0` and with `VoI > 0` both exist, so a "learn a stable regime" method can
activate on a predictable but economically inert partition.

**Corollary (abstention is Bayes-optimal when `VoI <= 0`).** If the oracle state
policy has non-positive value, then `LCB(VoI) <= 0` and the gate correctly collapses to
the fixed threshold.

Full proofs are in S1 Text. For a finite policy class of size `M` and bounded utility,
a Hoeffding bound with a union bound gives `|V_hat - V| <= B sqrt(2 log(2M/delta)/n)`
with probability `1-delta`; for dependent data we replace `n` by a block-dependent
effective sample size estimated by a moving-block bootstrap.

### Statistical analysis

Software: Python 3.13 with NumPy 2.5.0, pandas 3.0.3, scikit-learn 1.9.0, SciPy
1.18.0, and Matplotlib 3.11.0. The primary metric is the group-level net-utility lower
bound (`NU_LB`), the 5% quantile of a moving-block bootstrap of the annualized mean net
utility, with block length 10 and 2000 replications (seed 20260928); groups are
`asset x model x fold` and are averaged equally. Thresholds are selected on the
validation split only, over the grid 0.30 to 0.80 (51 points). Activation and
abstention rates are reported as exact counts with Wilson score 95% confidence
intervals. Decision regret and the conditional action value `Delta(p,z)` are estimated
in probability bins (10 bins; a 5/10/20-bin sensitivity gives the same correlation,
see Results). Sample sizes: Experiment A uses 8 synthetic scenarios x 20 seeds (160
rows) plus the frozen Bitcoin/Tesla bundle; Experiment B uses 8 seeds per scenario;
Experiment C uses 12 seeds per scenario; Experiment D uses 20 seeds for placebos and 10
bootstrap resamples per scenario for split stability. The significance threshold is
alpha = 0.05 and all intervals are two-sided; no multiplicity correction is applied
because the scenario comparisons are exploratory and the headline claims rest on the
controlled positive/negative contrast. All analyses are deterministic given the fixed
seeds and reproduce bit-identically. Analysis code is available in the accompanying reproduction package and the public
repository (see Data and Code Availability in the submission system).

## Results

### Experiment A — Decision regret, not calibration, tracks the gain

We estimated `Delta(p,z)` in probability bins and computed decision regret for fixed
versus state-dependent thresholds over 20 seeds x 8 DGP scenarios (160 rows), and
compared the regret reduction with ECE and Brier scores (Table 1).

**Table 1. Decision regret by scenario (mean over 20 seeds).**

| Scenario | regret reduction | state contrast | ECE | Brier | realized gain |
|---|---:|---:|---:|---:|---:|
| decision_relevant_state | +0.00361 | 0.0297 | 0.035 | 0.196 | +1.025 |
| asymmetric_utility | +0.00051 | 0.0119 | 0.037 | 0.197 | +0.156 |
| high_separation | +0.00005 | 0.0032 | 0.032 | 0.194 | +0.001 |
| low_separation | +0.00006 | 0.0032 | 0.035 | 0.199 | -0.001 |
| low_dispersion | +0.00005 | 0.0045 | 0.027 | 0.233 | +0.006 |
| weak_detector | +0.00005 | 0.0032 | 0.032 | 0.195 | +0.000 |
| uninformative_regime | +0.00004 | 0.0030 | 0.030 | 0.196 | -0.006 |
| no_transition | +0.00000 | 0.0000 | 0.034 | 0.198 | -0.011 |

Correlations over the 160 rows: regret reduction and realized gain **+0.945**; state
contrast and gain +0.935; **ECE and gain +0.098**; Brier and gain -0.129; ECE and
regret reduction +0.110. The correlation is stable across probability-bin counts (5/10/20
bins give +0.967/+0.968/+0.968; `bin_count_sensitivity_summary.csv`). The decision-regret object tracks the realized gain; the
calibration statistics do not (Fig 1). `high_separation` is statistically separated
but decision-irrelevant, so both its regret reduction and its gain are ~0.

**Fig 1. Decision-regret mechanism.** (A) Conditional action value `Delta(p,z)` for a
decision-relevant data-generating process (the two state curves cross zero at different
probabilities) and for a state-invariant process (curves coincide). (B) Decision regret
of a single fixed threshold; the gap between the fixed-threshold minimum and the
state-aware minimum is the value of information.

### Experiment B — Probability calibration and decision calibration are separate levers

We ran a 2x2 design (raw versus temperature-scaled probabilities x fixed versus RADC
thresholds). Because the calibrator is monotone, we mapped the threshold grid through
the same monotone map so the calibrated cell evaluates the identical decision set
(Proposition 4). Factorization of external `NU_LB` is given in Table 2.

**Table 2. Factorization of external `NU_LB`.**

| Effect | Synthetic | Real |
|---|---:|---:|
| probability calibration (within fixed) | +0.0000 | +0.0000 |
| decision calibration (within raw) | +0.2881 | 0.0000 |
| decision calibration (within calibrated) | +0.2881 | 0.0000 |
| interaction | +0.0000 | +0.0000 |
| ECE, raw to calibrated | 0.0343 to 0.0354 | 0.0742 to 0.0172 |

On real Bitcoin/Tesla, temperature scaling reduced ECE from 0.074 to 0.017 with
**zero** change in external utility. The two levers are additive and orthogonal:
probability calibration is decision-inert, decision calibration is not. Calibration
appears to matter only through a fixed absolute anchor such as 0.5: at the 0.5 anchor
both raw and calibrated `NU_LB` equal -0.4014, versus -0.0935 at the re-optimized
threshold. That gap is an anchoring artifact, not a calibration effect.

**Table 3. Sensitivity to the calibrator (effect on external `NU_LB`).** The
invariance in Proposition 4 requires *strict* monotonicity.

| Calibrator | Synthetic | Real |
|---|---:|---:|
| temperature (strict) | +0.00000 | +0.00000 |
| Platt (strict) | +0.00000 | +0.00000 |
| isotonic (non-strict) | -0.05763 | -0.06792 |

Isotonic regression has flat regions, so its mapped cutpoints are coarsened and it can
change (and here, lower) the utility. The decision-inertness result is therefore a
property of strict monotonicity, not of "calibration" in general.

### Experiment C — The gate recovers the oracle gain and abstains when value is negative

We computed a deployable oracle (per-true-state thresholds selected on the search
split), an optimistic in-sample ceiling, and the RADC realized gain, over 12 seeds per
scenario (Table 4).

**Table 4. Oracle upper bound and RADC recovery.**

| Scenario | oracle LB gain | RADC gain | recovery | activation |
|---|---:|---:|---:|---:|
| decision_relevant_state | +1.0129 | +1.0199 | 1.009 | 100% |
| asymmetric_utility | +0.1559 | +0.0518 | 0.312 | 33% |
| high_separation | +0.0038 | 0.0000 | 0.00 | 0% |
| low_separation | -0.0109 | 0.0000 | 0.00 | 0% |
| low_dispersion | +0.0075 | 0.0000 | 0.00 | 0% |
| weak_detector | +0.0047 | 0.0000 | 0.00 | 0% |
| uninformative_regime | -0.0131 | 0.0000 | 0.00 | 0% |
| no_transition | -0.0064 | 0.0000 | 0.00 | 0% |

Correlation between oracle lower-bound gain and RADC gain is **+0.987**. On the strong
positive control the estimator recovers essentially the entire oracle gain (Fig 2).

**Fig 2. Oracle upper bound and recovery.** (A) Realized RADC gain versus the oracle
lower-bound gain, with the y = x line; points on the line are full recovery.
(B) Oracle lower-bound gain versus captured gain by scenario.

**Table 5. Activation rates with exact counts and Wilson 95% intervals.**

| Case | Activated / runs | Rate | Wilson 95% CI |
|---|---:|---:|---|
| decision_relevant_state (strong positive) | 20/20 | 100% | [83.9%, 100%] |
| positive controls (pooled) | 16/27 | 59.3% | [40.7%, 75.5%] |
| asymmetric_utility (moderate) | 4/12 | 33.3% | [13.8%, 60.9%] |
| negative controls (pooled) | 0/69 | 0% | [0%, 5.3%] |
| abstention when oracle gain near zero | 65/65 | 100% | [94.4%, 100%] |

Negative-control activation is at most 5.3% at 95% confidence, and the gate abstains
on essentially every case with no positive oracle gain.

### Experiment D — Stability is not decision relevance

Destroying the state-outcome alignment with a circular time shift or a block
permutation removed the entire gain for the decision-relevant control and changed
nothing for the decision-irrelevant controls (Table 6).

**Table 6. State-shuffle placebo (mean over 20 seeds).**

| Scenario | real gain | circular shift | block permute |
|---|---:|---:|---:|
| decision_relevant_state | +1.0251 | -0.1327 | -0.0685 |
| asymmetric_utility | +0.1558 | -0.0921 | -0.1108 |
| high_separation | +0.0009 | +0.0018 | -0.0017 |
| uninformative_regime | -0.0064 | -0.0038 | -0.0018 |

Bootstrap split stability is not the discriminator: a stable split exists even for
`high_separation` (stable in 100% of resamples; mean bootstrap selection frequency
0.81) and `uninformative_regime` (feature selected in 8/10 seeds; mean frequency 0.67),
yet the decision gate abstains on both (activation 0%). This isolates the mechanism:
stability is necessary but not sufficient, and the value-of-information gate is the
operative criterion (Proposition 5).

### Real assets — the abstention certificate

On the frozen Bitcoin/Tesla bundle, with the detector posterior as an explicitly
labelled pseudo-oracle proxy (no true states exist), the upper bound is negative
(Table 7).

**Table 7. Real-asset boundary.**

| fixed `NU_LB` | pseudo-oracle `NU_LB` | pseudo-oracle gain | RADC gain | activated |
|---:|---:|---:|---:|---:|
| -0.0935 | -1.0169 | -0.9234 | 0.0000 | No |

There is no positive value of state information to capture, so RADC collapses to the
fixed threshold and does not degrade it (Fig 3). We regard this as the correct behavior of a
decision-calibration method and report it as a boundary condition.

**Fig 3. Mechanism confirmation across 160 synthetic rows.** (A) Oracle regret
reduction versus realized external gain. (B) Regret reduction (blue) versus expected
calibration error (red) by scenario; calibration error is flat and does not track the
gain.

## Discussion

**Why rule-based regime gating failed.** It conditions the decision on a partition
chosen for statistical properties, not for decision value. A statistically stable
partition can be economically inert (Proposition 5), and probability-calibration
improvements are decision-inert for re-optimized thresholds (Proposition 4). The regime
threshold in our reassessment lost because it churned and traded on decision-irrelevant
states, whereas the fixed threshold captured the same directional signal at lower cost.

**What RADC changes.** It treats "should the decision depend on the state?" as the
object to be estimated. It learns a context partition, tests it under resampling, and
activates state-dependent thresholds only when a lower confidence bound on the value of
state information clears a risk buffer; otherwise it collapses to the best fixed
threshold, which is Bayes-optimal when the value is non-positive.

**Limitations.** RADC is deliberately conservative and activated in only 33% of runs on
a moderate control. A cost-robust, decision-relevant regime is required for activation,
and in the current Bitcoin/Tesla data no such regime exists. This is desirable for a
decision-calibration method but limits immediate deployment. Our real-asset evidence
uses a frozen two-asset testbed; broader asset coverage is left to future work. The
synthetic controls identify the mechanism but are not evidence of real-market gains.

**Conclusion.** We reframed regime-aware thresholding as Regime-Aware Decision
Calibration. The utility gain of conditioning a threshold on a state equals the
reduction in decision regret minus cost, not an improvement in probability calibration
or regime classification. We formalized when state information can be ignored, and
built a two-gate estimator that must earn the right to use regimes. Controlled
experiments confirm the mechanism; on the current Bitcoin/Tesla data the oracle upper
bound is negative and the method correctly abstains. The open empirical question is
whether any asset contains a cost-robust, decision-relevant regime; the theory and the
estimator make that question well-posed.

## Acknowledgments

The author thanks the maintainers of the public data sources used in this study (Yahoo Finance; blockchain.com; lookintobitcoin.com) and the maintainers of the open-source scientific software (NumPy, pandas, scikit-learn, SciPy, Matplotlib).

## References

All entries below are either well-known classics (verifiable metadata) or papers in the
project's downloaded literature set, whose identifiers were extracted from
`literature/metadata/*.json` and `literature/summaries/*`. No identifier is invented.

1. Hamilton JD. A new approach to the economic analysis of nonstationary time series
   and the business cycle. Econometrica. 1989;57(2):357-384.
2. Ang A, Timmermann A. Regime changes and financial markets. Annual Review of
   Financial Economics. 2012;4:313-337.
3. Zhang et al. RegimeFolio: a regime aware ML system for sectoral portfolio
   optimization in dynamic markets. arXiv:2510.14986. 2025.
4. RegimeNAS: regime-aware differentiable architecture search with theoretical
   guarantees for financial trading. arXiv:2508.11338. 2025.
5. Adaptive financial transformer with regime-gated attention for stock return
   prediction. arXiv:2606.29347. 2026.
6. Adaptive and regime-aware reinforcement learning for portfolio optimization.
   arXiv:2509.14385. 2025.
7. The optimal threshold selection for high-frequency pairs trading via supervised
   machine learning algorithms. Computational Economics. 2025.
   doi:10.1007/s10614-025-10958-5.
8. Forecasting economically significant Bitcoin moves: a multi-scale TCN with
   profit-optimized thresholds. arXiv:2608.26174. 2026.
9. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks.
   Proceedings of the 34th International Conference on Machine Learning (ICML). 2017.
   arXiv:1706.04599.
10. Niculescu-Mizil A, Caruana R. Predicting good probabilities with supervised
    learning. Proceedings of the 22nd International Conference on Machine Learning
    (ICML). 2005.
11. Robust decision making with partially calibrated forecasts. arXiv:2510.23471. 2025.
12. Calibrating decision robustness via inverse conformal risk control.
    arXiv:2510.07750. 2025.
13. Inference-time decision calibration for temporal classification. arXiv:2606.16034.
    2026.
14. Forecast skill is not decision skill: evidence from weather-dependent decision
    tasks. arXiv:2512.14779. 2025.
15. Dynamic regime-aware conformal calibration for reliable economic forecast
    intervals under multiple distribution shifts. arXiv:2608.17079. 2026.
16. Howard RA. Information value theory. IEEE Transactions on Systems Science and
    Cybernetics. 1966;2(1):22-26.
17. Vovk V, Gammerman A, Shafer G. Algorithmic Learning in a Random World. Springer;
    2005.
18. Fischer T, Krauss C. Deep learning with long short-term memory networks for
    financial market predictions. European Journal of Operational Research.
    2018;270(2):654-669.
19. Bailey DH, Borwein JM, Lopez de Prado M, Zhu QJ. The probability of backtest
    overfitting. Journal of Computational Finance. 2017;20(4):39-69.
20. Lopez de Prado M. Advances in Financial Machine Learning. Wiley; 2018.
21. What survives honest evaluation? Leakage-safe, search-aware assessment of
    LLM-driven trading strategy discovery. arXiv:2608.27734. 2026.
22. Equity strategy backtesting: luck or edge? The MinervaScore as a statistical
    robustness grade. arXiv:2608.23808. 2026.
23. Machine learning-based Bitcoin trading under transaction costs: evidence from
    walk-forward forecasting. arXiv:2606.00060. 2026.
24. Dynamic feature selection with distributional Markov chain error correction for
    financial time series forecasting. Applied Soft Computing. 2026.
    doi:10.1016/j.asoc.2026.114906.
25. Regime-aware deep learning for financial forecasting: an adaptive approach.
    Expert Systems with Applications. 2026. doi:10.1016/j.eswa.2026.133584.


## Supporting information

- **S1 Text. Proofs of Propositions 1-5 and the finite-sample bound.** Full proofs,
  including the strictness caveat for Proposition 4 and the isotonic counterexample.
- **S2 Text. Protocol and split definitions.** `configs/evaluation_protocol.json`,
  `configs/data_splits.json`.
- **S1 Data. Frozen prediction bundle and result tables.** `predictions_v1.csv`,
  `artifacts/results/*`, and the SHA-256 manifest.
- **S2 Data. Supplementary tables.** `activation_rates.csv`,
  `calibration_method_sensitivity.csv`, `bin_count_sensitivity_summary.csv`.
- **S1 Code. Analysis code.** `src/rutl/`, `experiments/`, `tests/`.
- **S1 Fig. Decision-regret mechanism.** File `Fig1.png`.
- **S2 Fig. Oracle upper bound and recovery.** File `Fig2.png`.
- **S3 Fig. Mechanism confirmation.** File `Fig3.png`.
