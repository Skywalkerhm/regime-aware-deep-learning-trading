# Regime-Aware Deep Learning for Bitcoin Trading

Cross-model study of regime-aware decision threshold modulation for Bitcoin and TSLA trading strategies.

**Final paper:** paper_jfds_v4.docx (also available as paper_jfds.md)

## Key Results

| Model | Pure Sharpe | +Regime | Delta | Calibration (ECE) |
|------|-----------|--------|-------|-------------------|
| Transformer | +0.253 | **+1.332** | +1.079 | 0.098 (worst) |
| LSTM | +0.792 | **+1.520** | +0.728 | 0.076 |
| XGBoost | +0.993 | +0.540 | -0.453 | 0.030 (best) |

- Cross-asset validation on TSLA confirms the effect (LSTM: -0.229 to +1.848, 8/8 folds)
- Bootstrap test p=0.0001, 95% CI [+0.322, +1.168]; Bonferroni-corrected for k=5
- ECE monotonic ordering (Transformer > LSTM > XGBoost) supports the calibration mechanism

## Reproduction

pip install -r requirements.txt
python build_dataset.py
python btc_conditional/train_all_folds.py
python btc_conditional/validate_hybrid.py
python btc_conditional/regime_final.py
python btc_conditional/evaluate_calibration.py
python btc_conditional/baseline_xgboost.py
python btc_conditional/train_transformer.py
python btc_conditional/validate_tsla.py
python verify.py

## File Structure

paper_jfds_v4.docx          Final paper (Word)
paper_jfds.md               Final paper (Markdown)
btc_conditional/             Source code
  model.py, trainer.py       LSTM model + training
  validate_hybrid.py         Main validation
  evaluate_calibration.py    ECE/Brier calibration
  results_v2/                Result CSV files
verify.py                    Consistency check

## Data Sources

Bitcoin OHLCV: Yahoo Finance (2014-2026)
TSLA OHLCV: Yahoo Finance (2010-2026)
On-chain: blockchain.com, lookintobitcoin.com
