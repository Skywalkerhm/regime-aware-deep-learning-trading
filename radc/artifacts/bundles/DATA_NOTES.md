# Data notes

`predictions_v1.csv` is the **derived** bundle used by all analyses: 36,690 rows x 19
columns of model outputs (`p`, `u`, `posterior_*`, `uncertainty`), derived quantities
(`r_next`, `y`, `return_20d`, `volatility_20d`, `trend_strength`), and identifiers.
No raw vendor price or on-chain series are included.

Raw sources (third-party; not redistributed): Yahoo Finance (BTC-USD, TSLA),
blockchain.com (on-chain metrics), lookintobitcoin.com (NUPL/CDD/MVRV/Fear&Greed).
`build_dataset.py` in the repository regenerates the features from the public sources.

SHA-256: `b4d8253fada74e5b9bd5e1bdae9c1f6e6d43356d2731260935c2cafbe22f19e9`  predictions_v1.csv
