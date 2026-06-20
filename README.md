# Regime-Aware Deep Learning for Bitcoin Trading

Cross-model study of regime-aware decision threshold modulation for Bitcoin and TSLA trading strategies.

## Key Results

| Model | Pure Sharpe | +Regime | Delta | Calibration (ECE) |
|------|-----------|--------|-------|-------------------|
| Transformer | -0.253 | **+1.332** | +1.079 | 0.098 (worst) |
| LSTM | +0.792 | **+1.520** | +0.728 | 0.076 |
| XGBoost | +0.993 | +0.540 | -0.453 | 0.030 (best) |

- Cross-asset validation on TSLA confirms the effect (LSTM: -0.229 to +1.848, 8/8 folds)
- Bootstrap test p=0.0001, 95% CI [+0.322, +1.168]; Bonferroni-corrected for k=5

## Paper

- paper_jfds.md — JFDS submission draft (Markdown)
- paper_jfds.docx — Word version

## Reproduction

`ash
git clone https://github.com/Skywalkerhm/regime-aware-deep-learning-trading.git
cd regime-aware-deep-learning-trading
pip install -r requirements.txt

# BTC data + 8-fold LSTM training
python build_dataset.py
python btc_conditional/train_all_folds.py

# Run main results
python btc_conditional/validate_hybrid.py        # BTC 8-fold hybrid validation
python btc_conditional/regime_final.py            # 5-strategy comparison
python btc_conditional/evaluate_calibration.py    # ECE/Brier for all 3 models
python btc_conditional/baseline_xgboost.py        # XGBoost baseline
python btc_conditional/train_transformer.py       # Transformer baseline

# TSLA
python btc_conditional/validate_tsla.py           # TSLA cross-asset validation

# Consistency check
python verify.py
`

## File Structure

`
btc_conditional/
  model.py, model_transformer.py           # Model architectures
  trainer.py, backtest.py, metrics.py      # Core engine
  validate_hybrid.py, regime_final.py      # Main validations
  evaluate_calibration.py                  # ECE/Brier calibration
  baseline_xgboost.py, train_transformer.py # Baselines
  validate_tsla.py                         # TSLA cross-asset
  results_v2/                              # All result CSVs

data/processed/                              # BTC + TSLA features
paper_jfds.md, paper_jfds.docx               # Paper
`

## Data Sources

- Bitcoin OHLCV: Yahoo Finance (2014-2026)
- TSLA OHLCV: Yahoo Finance (2010-2026)
- On-chain metrics: blockchain.com, lookintobitcoin.com

## Methodology

- **Regime detection**: 4-state classifier (trend_up/trend_down/chop/crisis) using volatility and moving-average signals
- **Walk-forward**: 8 folds, 2-year train / 6-month test
- **Transaction cost**: 10 bps on position changes
- **Statistical tests**: Bootstrap (10,000 resamples), paired t-test, Wilcoxon signed-rank, Bonferroni correction

## License

[MIT](LICENSE) or add your preferred license.
