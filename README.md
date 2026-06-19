# BTC Regime-Aware Trading Strategy

A market regime-aware Bitcoin trading framework combining LSTM prediction with real-time regime gating.

## Core Mechanism

- **Regime Gate**: Classifies market state (trend_up/trend_down/chop/crisis) using only real-time observable price features
- **LSTM Controller**: Two-layer LSTM predicting intraday return direction, outputting probability as position size
- **Hybrid**: Regime decides when NOT to trade, LSTM decides how much to trade

## Key Result

**Hybrid strategy achieves Sharpe +1.520** (p=0.016) across 8 walk-forward folds (2018-2026), significantly outperforming standalone LSTM (+0.792, p=0.016) and Buy & Hold (+1.108). Hybrid wins 7/8 folds. An ablation removing same-day open price confirms the results are robust (Hybrid +1.585 vs LSTM +0.953).

## Setup

```bash
pip install -r requirements.txt
python3 build_dataset.py              # Build data pipeline
python3 btc_conditional/train.py      # Train 8-fold LSTM models
```

## Files

| File | Purpose |
|------|---------|
| `build_dataset.py` | Data pipeline (Kaggle + Yahoo → merged CSV) |
| `btc_conditional/model.py` | Two-layer LSTM |
| `btc_conditional/trainer.py` | Trainer + data loader + grid search |
| `btc_conditional/backtest.py` | Backtest engine (10bps cost) |
| `btc_conditional/metrics.py` | Performance metrics |
| `btc_conditional/regime_final.py` | Final regime strategy comparison |
| `btc_conditional/continuous_all.py` | Continuous position sizing comparison |
| `btc_conditional/run_ablation.py` | Feature ablation |
| `btc_conditional/analyze_all.py` | Comprehensive statistical analysis |

## Data Sources

- Price: Yahoo Finance (2014-2026)
- On-chain: Kaggle (blockchain.com + lookintobitcoin, 2009-2023)
- Backup: `data/raw_backup/` contains all original files

## Leakage Audit

All potential look-ahead biases have been checked and sealed:

1. Regime thresholds calibrated on training data only (2014-2018)
2. Backtest profit aligned to predicted return (return[t+1], not return[t])
3. Hybrid coefficients frozen to default values (no tuning)
4. Fold labels not used by any strategy
5. Feature timestamps audited (delayed chain features don't affect regime)
6. All statistical tests include bootstrap CI and paired tests
