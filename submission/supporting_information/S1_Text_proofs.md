# Formal Theory: Decision-Relevant Regimes and Value-of-Information Gating

## Notation

Let:

- `X` be context;
- `P ∈ [0,1]` be a forecast probability;
- `Z` be a latent decision state;
- `A ∈ {0,1}` be long/flat action;
- `R` be next-period return;
- `U(a, r)` be bounded net utility after transaction costs and drawdown penalties;
- `θ_f` be a fixed threshold;
- `θ*(z)` be the optimal threshold conditional on state `z`.

Define the fixed-policy value:

```text
V_fixed(θ) = E[U(1[P >= θ], R)].
```

Define the oracle state-policy value:

```text
V_state(θ*(.)) = E[U(1[P >= θ*(Z)], R)].
```

Define the value of information:

```text
VoI* = V_state(θ*(.)) - max_θ V_fixed(θ).
```

## Assumption A1: Bounded utility

There exists `B < ∞` such that:

```text
|U(a,r)| <= B
```

for all feasible actions and returns.

## Assumption A2: Lipschitz threshold value

For each state distribution `D_z`, the map:

```text
θ -> E[U(1[P >= θ], R) | Z=z]
```

is `L_z`-Lipschitz, with `L = sup_z L_z < ∞`.

A sufficient condition is that the conditional density of `P` is bounded and the utility is bounded.

## Assumption A3: State approximation

A learned state representation `G = g(X)` and estimated threshold map `θ_hat(G)` are available. Define:

```text
E_state = E|θ_hat(g(X)) - θ*(Z)|
```

on the target distribution.

## Proposition 1: Decision-relevance decomposition

Under A1–A3:

```text
V_state(θ_hat) >= V_state(θ*) - L * E_state.
```

If `θ_hat(g(X))` is decomposed as the oracle restricted to the learned state plus
a threshold-estimation error, then:

```text
V_state(θ_hat) >= V_state(θ*) 
    - L * E_oracle_state_approximation
    - L * E_threshold_estimation.
```

### Proof sketch

For every realized context and state, apply Lipschitzness to the threshold
mapping. Take expectations over the target distribution and use the triangle
inequality to separate state approximation from threshold estimation. Boundedness
A1 ensures the expectations are finite and the interchange of limits is valid
under standard integrability conditions.

## Proposition 2: No-regime collapse

Suppose for all states in the support of `Z`:

```text
θ*(z) = θ_f* = argmax_θ V_fixed(θ).
```

Then:

```text
VoI* = 0.
```

Furthermore, if the VoI estimator is conservative and subtracts non-negative
complexity and uncertainty penalties, the activation condition:

```text
LCB(VoI_hat) > 0
```

cannot be satisfied.

### Proof

The state policy applies the same threshold `θ_f*` for every state, so its
value equals the fixed-policy value. Any non-negative penalty can only lower the
estimated VoI. Therefore the lower confidence bound is non-positive.

### Interpretation

Failure to activate is not a method failure. It is the correct behavior when
state labels have no decision value.

## Proposition 3: State-relevance separation condition

Fix two states `z1` and `z2` and suppose their optimal thresholds satisfy:

```text
θ*(z1) < θ*(z2).
```

Let the conditional probability density of `P` be bounded below by `m > 0` on
an interval `I ⊂ (θ*(z1), θ*(z2))`, and let `α = min(P(Z=z1), P(Z=z2)) > 0`.
If the utility difference between the two state-optimal actions is strictly
positive on `I`, then the oracle state policy has strictly greater utility than
the best fixed policy.

### Proof sketch

On `I`, the actions induced by the two state-optimal thresholds differ. Because
the density has positive mass there and both states occur with positive
probability, the policy that uses the state information captures a strictly
positive utility gain relative to any single common threshold. Taking `I` to be
the region where the action disagreement has positive expected utility
establishes strict separation.

This condition formalizes the empirical finding that regime labels can exist
without decision relevance. Statistical separation of state distributions is
not enough; the optimal action must change on a probability region with positive
economic value.

## Proposition 4: Finite-sample gate

Let the policy class have finite cardinality `M`. Under A1 and an i.i.d.
validation sample of size `n`, for any `δ ∈ (0,1)`, with probability at least
`1-δ`:

```text
|V_hat(π) - V(π)| <= B sqrt(2 log(2M/δ)/n)
```

for all policies in the class.

### Proof

Apply Hoeffding's inequality to each policy and a union bound over `M`
policies.

For dependent or nonstationary financial data, replace the effective sample
size `n` by a block-dependent effective sample size and estimate the lower
confidence bound using a moving-block bootstrap. This motivates the empirical
stability gate and VoI lower bound used by DRRL-VoI.

## Corollary: Activation requires value above risk

If the gate also requires a risk buffer `R_alpha` estimated by split conformal
quantiles, the activation condition becomes:

```text
LCB(VoI_hat) > R_alpha.
```

The resulting method uses state-dependent thresholds only when the estimated
economic value exceeds both sampling uncertainty and downside-risk uncertainty.

## Remaining formal work

1. State a precise β-mixing or block-dependence condition.
2. Replace the finite-class bound with a VC or Rademacher bound.
3. Prove consistency of the stability-selected split under a resampling model.
4. Connect the split-conformal risk buffer to a finite-sample coverage guarantee.

---

## Decision-calibration extension (2026-10-09)

The results in `artifacts/reports/decision_regret_report.md` require two further
formal statements: (i) the value of information is a *decision-regret* object, and
(ii) probability calibration cannot certify it.

### Notation addendum

Work with the gross long/flat action value `U(1, r) = r`, `U(0, r) = 0`, and define

```text
Delta(p, z) = E[R | P=p, Z=z]      (conditional utility gap of holding)
a*(p, z)    = 1[Delta(p, z) >= 0]  (Bayes action)
```

Decision regret of a threshold rule `theta(z)`:

```text
R_dec(theta) = E[ max(0, -Delta(P,Z)) * 1[P >= theta(Z)]
                + max(0,  Delta(P,Z)) * 1[P <  theta(Z)] ].
```

### Proposition 5 (VoI is a decision-regret object, not a calibration object)

Let `theta_f` minimize `R_dec` over fixed thresholds and `theta_z` over
state-dependent thresholds. Then

```text
V_state(theta_z) - V_fixed(theta_f) = R_dec(theta_f) - R_dec(theta_z) - C(theta_z),
```

where `C(theta_z) >= 0` collects the expected transaction, turnover, and drawdown
costs that the gross regret omits.

**Proof.** Write the expected utility of any rule as
`V(theta) = E[ a * Delta ] - C(theta)`. The Bayes value
`E[ max(Delta, 0) ] = V(theta) + R_dec(theta)` holds for every rule, because
`R_dec` is exactly the expected shortfall against `a*`. Hence
`V(theta) = E[max(Delta,0)] - R_dec(theta) - C(theta)`, and subtracting the fixed
rule gives the identity. `square`

**Consequence.** Maximizing utility is equivalent to minimizing regret minus cost.
The state information is decision-relevant iff it strictly reduces `R_dec` on a
positive-probability set; it is economically usable iff the reduction exceeds
`C(theta_z)`.

### Proposition 6 (probability calibration does not certify decision relevance)

Let `P` be a probability-calibrated forecast, `P(Y=1|P=p) = p` for all `p`. The
value of information can still be zero or negative under a state-dependent payoff,
and arbitrarily large under another DGP that induces the *same* calibrated marginal
`P`.

**Proof.** ECE and reliability constrain only the marginal
`P(Y=1|P=p)`. `Delta(p,z) = E[R|P=p,Z=z]` also depends on the conditional payoff
asymmetry, which is not identified by `P(Y=1|P=p)`. Construction: take two states
with equal `P(Y=1|P=p)=p`. Set payoffs

```text
DGP A: state-invariant     gain = loss = 0.012
DGP B: state-0 gain 0.035, loss 0.005 ; state-1 gain 0.005, loss 0.035
```

Both share identical calibrated marginal probabilities and therefore identical ECE
and Brier scores, but `Delta(p, z)` is state-invariant in A (VoI = 0) and
state-separated in B (VoI > 0). Since the payoff parameters are free, the VoI gap is
unbounded. Therefore no probability-calibration statistic can certify decision
relevance. `square`

**Empirical confirmation.** Over 160 synthetic rows, ECE ~ realized gain has
correlation +0.098 whereas decision-regret reduction ~ realized gain has correlation
+0.945. `high_separation` and `decision_relevant_state` have comparable ECE
(0.032 vs 0.035) but regret reductions differing by ~70x and realized gains
differing by ~1000x.

### Proposition 7 (state-relevance separation, refined)

For a refinement `z -> (z, z')` to strictly improve the Bayes decision value,
there must exist a set `S` of positive probability with

```text
a*(p, z) != a*(p, z')  on S,   and   E[ |Delta(P,Z)| * 1_S ] > C(theta_z).
```

**Proof sketch.** Action disagreement on `S` is necessary for the two state
policies to differ; without it they coincide and the value difference is zero. Given
disagreement, the regret reduction equals the expected magnitude of the avoided
mistakes on `S`, which must dominate the added cost `C(theta_z)` for a strict gain.
`square`

The first condition is exactly the *decision*-relevance condition; it is strictly
stronger than statistical separability of the state-conditional forecast
distributions, which is insufficient (see Proposition 6 and the `high_separation`
control).

### Remaining formal work

1. Replace the i.i.d. finite-class gate with a block-dependence (beta-mixing)
   Rademacher bound (Proposition 4 extension).
2. Prove consistency of the stability-selected split under the resampling model.
3. Derive a coverage guarantee for the split-conformal risk buffer used by the
   activation gate.

---

## Monotone-calibration invariance (2026-10-09)

Experiment B (`experiments/calibration_decision_2x2.py`) requires one more
statement: the usual probability calibrators cannot move a threshold decision.

### Proposition 8 (strictly monotone recalibration is decision-inert)

Let `phi: [0,1] -> [0,1]` be *strictly* increasing. For any threshold family
`{1[p >= theta] : theta in Theta}` with `Theta ⊆ [0,1]`,

```text
{ 1[phi(p) >= phi(theta)] : theta in Theta } = { 1[p >= theta] : theta in Theta }.
```

Therefore, for any utility, the set of achievable action vectors -- and hence the
optimal expected utility over threshold rules -- is invariant under any strictly
monotone recalibration. Temperature scaling and Platt scaling (strictly monotone)
cannot change the value of a re-optimized threshold rule.

**Proof.** Since `phi` is strictly increasing, `phi(p) >= phi(theta) <=> p >= theta`.
Mapping `theta -> phi(theta)` is a bijection from `Theta` onto `phi(Theta)`, so the
two action families coincide pointwise. `square`

### Remark (non-strict monotonicity breaks invariance)

If `phi` is only non-decreasing, then

```text
1[phi(p) >= phi(theta)] = 1[p >= tau(theta)],  tau(theta) = inf{p : phi(p) >= phi(theta)} >= theta.
```

The mapped grid `phi(Theta)` therefore realizes the *coarsened* cutpoint family
`tau(Theta)`, which is generally a strict subset of `Theta`. Isotonic regression is
monotone but not strictly monotone (it has flat regions), so it can change the
threshold decision -- empirically it *degrades* the re-optimized utility.

**Empirical check** (`experiments/calibration_method_sensitivity.py`,
`artifacts/results/calibration_method_sensitivity.csv`):

| Calibrator | effect on external NU_LB (synthetic) | (real BTC/TSLA) |
|---|---:|---:|
| temperature (strict) | +0.00000 | +0.00000 |
| Platt (strict) | +0.00000 | +0.00000 |
| isotonic (non-strict) | -0.05763 | -0.06792 |

The invariance is therefore a property of *strict* monotonicity, not of
"calibration" in general. This sharpens the claim and is reported as a caveat.

### Corollary (ECE is a decision-irrelevant statistic)

If a recalibration reduces ECE while remaining monotone, it leaves the decision
value unchanged:

```text
max_theta E[U(1[phi(P) >= phi(theta)], R)] = max_theta E[U(1[P >= theta], R)].
```

ECE improvement therefore cannot, by itself, certify any decision-layer gain.

### Where calibration *appears* to matter

Calibration changes decisions only when thresholds are anchored in absolute
probability units rather than re-optimized. The common anchor `theta = 0.5` is the
clearest example: applied to raw vs calibrated probabilities it selects different
quantiles and thus different actions. This is an *anchoring artifact*, not
economics, and it is the formal reason the earlier manuscript's use of 0.5 as a
benchmark was not a decision-relevant comparison.

### Empirical confirmation (Experiment B)

| Lever | synthetic effect on NU_LB | real effect on NU_LB |
|---|---:|---:|
| probability calibration (temperature) | +0.0000 | +0.0000 |
| decision calibration (DRRL-VoI) | +0.28 | 0.00 (no activation) |
| interaction | 0.0000 | 0.0000 |

On real BTC/TSLA, temperature scaling cut ECE from 0.074 to 0.017 with **zero**
change in external NU_LB, exactly as the invariance proposition predicts.

---

## Stability, placebo, and the abstention guarantee (2026-10-09)

Experiments C and D require two closing statements.

### Proposition 9 (stability does not imply decision relevance)

Let `G` be a state partition that is statistically stable under resampling: its
estimate `g_hat` converges to a fixed partition `g` as the sample grows. Stability
concerns only the *identification* of `g`; it says nothing about the sign or size of
the value of information. In particular, there exist stable partitions with
`VoI* = 0` (state-invariant optimal thresholds) and stable partitions with
`VoI* > 0`.

**Proof.** Stability is a property of the estimator distribution over the partition
space, whereas `VoI*` is a functional of `Delta(p, z) = E[R | P=p, Z=z]`. Choose any
partition `g` with full rank identification; define two payoff laws with identical
`P(Y=1|P,Z)` but different conditional `Delta`. Partition stability is identical
across the two laws, but `VoI*` differs. `square`

**Consequence.** A "learn a stable regime" method can activate on a statistically
real, decision-irrelevant partition. Only a decision-value gate can reject it.

### Corollary (abstention is Bayes-optimal when VoI <= 0)

If the oracle state policy has non-positive value over the best fixed threshold,
`VoI* <= 0`, then for every activation threshold `tau >= 0` the gate

```text
activate  iff  LCB(VoI_hat) > tau
```

is correct to abstain, since `VoI_hat` is a lower confidence bound on a
non-positive quantity. Collapsing to the fixed threshold is then optimal.

### Empirical confirmation

| Check | Result |
|---|---|
| placebo gain destroyed (decision_relevant_state) | +1.025 -> -0.13 (circular), -0.07 (block) |
| placebo gap (decision-irrelevant controls) | ~0 |
| stable split on high_separation | feature rate 0.81 (**stable**) |
| DRRL activation on high_separation | **0%** (gate rejects) |
| DRRL activation on negative controls | **0%** |
| abstention when oracle gain near zero | **100%** |
| real pseudo-oracle gain | **-0.92** (no positive bound) -> abstain |

### Summary statement of the mechanism

Regime-Aware Decision Calibration consists of two coupled gates:

```text
1. identification gate : is the state partition stable under resampling?
2. decision gate       : does LCB(VoI) exceed the risk buffer?
```

Both must pass. Partition stability (gate 1) is necessary but not sufficient;
the value-of-information condition (gate 2) is the operative decision-calibration
criterion. This two-gate structure is what distinguishes the approach from both
rule-based regime switching and probability calibration.
