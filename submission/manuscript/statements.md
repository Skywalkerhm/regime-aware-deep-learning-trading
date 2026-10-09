# Required Statements (paste-ready)

Author: **Min Huang**, The Chinese University of Hong Kong, Shenzhen, Guangdong,
China — minhuang1@link.cuhk.edu.cn, ORCID 0009-0007-4991-3221.

**No DOI is minted.** The data are shared as Supporting Information (archived by PLOS
in figshare) and the code via a public GitHub release, both of which satisfy the PLOS
Data Availability and Materials/Software/Code Sharing policies. Every field below is
final; the only optional action is pinning the GitHub URL to a tagged release (v1.0).

---

## Data Availability Statement

> The minimal data set required to replicate all findings is provided as **Supporting
> Information (S1 Data: `public_bundle_v1.csv`, 36,690 rows; S2 Data: result tables)**.
> Supporting Information is archived by PLOS in figshare upon publication. The same
> files are also available in the public repository
> **https://github.com/Skywalkerhm/regime-aware-deep-learning-trading** (release tag
> **v1.0**, license MIT; directory `reproduce/`).
>
> The deposited bundle (`public_bundle_v1.csv`) contains only derived quantities and
> model outputs: base-model probabilities `p` and `u`, regime posteriors, `uncertainty`,
> `volatility_20d`, `trend_strength`, `return_20d`, labels `y` and `r_next`, and the
> identifiers/splits. No raw vendor series are redistributed.
>
> The raw Bitcoin and Tesla price series are third-party data obtained from **Yahoo
> Finance** (BTC-USD; TSLA). Bitcoin on-chain metrics were obtained from
> **blockchain.com**, and valuation/sentiment indicators (Net Unrealized Profit/Loss,
> Coin Days Destroyed, MVRV/SOPR-type metrics, and the Fear and Greed Index) from
> **lookintobitcoin.com**. The authors do not hold rights to redistribute these raw
> series; the repository includes the download and feature-construction scripts
> (`build_dataset.py`) so that any researcher can obtain the data from the same public
> sources in the same manner and regenerate the bundle.

## Code Availability Statement

> All code is available at
> **https://github.com/Skywalkerhm/regime-aware-deep-learning-trading** (release tag
> **v1.0**, license **MIT**) and is mirrored in Supporting Information (S1 Code). It
> includes the RADC estimator and metrics (`src/rutl/`), the experiment runners
> (`experiments/`), and an automated test suite covering schema, causality, the
> evaluator, metrics, decision regret, calibration invariance, oracle/stability, and
> activation rates (`tests/`). `reproduce/run_all.sh` regenerates every table and
> figure from the frozen bundle. GitHub is an accepted open software archive under the
> PLOS Materials, Software and Code Sharing policy.

## Funding Statement

> "The author received no specific funding for this work."

---

## Competing Interests Statement

> "The author has declared that no competing interests exist."

---

## Author Contributions Statement (CRediT)

> Min Huang is the sole author and performed all roles:
>
> **Conceptualization, Methodology, Software, Validation, Formal analysis,
> Investigation, Data curation, Writing – original draft, Writing – review & editing,
> Visualization, Project administration:** Min Huang.

---

## Ethics Statement

> "This study did not involve human participants, human specimens, or animals. It uses
> only publicly available financial market data."

---

## Acknowledgments

> The author thanks the maintainers of the public data sources used in this study
> (Yahoo Finance; blockchain.com; lookintobitcoin.com) and the maintainers of the
> open-source scientific software (NumPy, pandas, scikit-learn, SciPy, Matplotlib).
> Do not list funding sources here (PLOS policy).

---

## Prior publication / duplicate submission

> "This manuscript is original, has not been published previously, and is not under
> consideration for publication elsewhere. No preprint of this work has been posted."

*(PLOS ONE requires a statement that the submission is not under consideration
elsewhere; there is no requirement to disclose prior editorial decisions at other
journals, so none is included.)*
