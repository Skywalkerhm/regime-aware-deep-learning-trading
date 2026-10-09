# Regime-Aware Decision Calibration

Code, data, and manuscript materials for **"Regime-aware decision calibration: when
should state information change a trading threshold?"** (PLOS ONE submission).

The repository also retains the earlier development material (see *Historical material*
below).

## The paper in one paragraph

Conditioning a trading threshold on a detected market "regime" is a popular, model-agnostic
intervention, but it often fails out of sample. We show that the objects usually invoked
to justify it — probability calibration and regime-classification accuracy — are not the
decision mechanism. The relevant object is **decision calibration**: a state partition
improves decisions only if it changes the optimal action on a region whose expected
utility exceeds the estimation and risk costs. We formalize this as a decision-regret /
value-of-information problem, define **Regime-Aware Decision Calibration (RADC)**, prove
five results, and give a two-gate estimator that activates state-dependent thresholds only
when a lower confidence bound on the value of state information exceeds a conformal risk
buffer, and otherwise collapses to the best fixed threshold.

## Repository layout

```
radc/                    Reproducible workspace for the paper
├── run_all.sh           One-command reproduction (verify -> test -> re-run -> diff)
├── artifacts/bundles/   Frozen derived prediction bundle + SHA-256
├── artifacts/results/   All result tables
├── configs/             Evaluation protocol and data splits
├── src/rutl/            RADC estimator, metrics, decision-regret module
├── experiments/         Experiment runners
├── tests/               Automated test suite
└── direction_evolve2/   DRRL configuration used by the calibration experiment
submission/              Manuscript materials
├── manuscript/          Main text, title page, cover letter, statements
├── figures/             Fig1.png, Fig2.png, Fig3.png
└── supporting_information/
LICENSE                  MIT
```

## Reproduce

```bash
cd radc
bash run_all.sh
```

This verifies the derived-bundle SHA-256, runs the test suite, re-runs all experiments into
`out/`, and diffs them against `artifacts/results/` (differences should be exactly 0).

Environment: Python 3.13, NumPy 2.5.0, pandas 3.0.3, scikit-learn 1.9.0, SciPy 1.18.0,
Matplotlib 3.11.0 (`radc/environment.txt`).

## Data

`radc/artifacts/bundles/predictions_v1.csv` is a **derived** bundle (36,690 rows x 19
columns): base-model probabilities (`p`, `u`), regime posteriors, `uncertainty`,
`volatility_20d`, `trend_strength`, `return_20d`, labels `y` and `r_next`, and the
identifiers/splits. It contains **no raw vendor series**.

Raw data are third-party and are **not redistributed**: Bitcoin and Tesla daily prices from
**Yahoo Finance** (BTC-USD, TSLA); Bitcoin on-chain metrics from **blockchain.com**;
valuation/sentiment indicators (NUPL, Coin Days Destroyed, MVRV/SOPR-type, Fear & Greed)
from **lookintobitcoin.com**. `build_dataset.py` regenerates the features from these public
sources.

## License

Code: **MIT** (see `LICENSE`). Data and results: CC BY 4.0.

## Historical material

`btc_conditional/`, `paper_jfds*.md`, `paper.md`, `data/`, `verify.py` and
`RESEARCH_REPORT.md` are from the earlier development of this research line and are kept
for provenance.
