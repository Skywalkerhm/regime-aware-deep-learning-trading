"""verify.py — 全量一致性检查"""
import pandas as pd, numpy as np, sys

errors = 0

# 1. Check regime_final vs continuous_all hybrid consistency
rf = pd.read_csv('btc_conditional/results_v2/regime_final.csv')
ca = pd.read_csv('btc_conditional/results_v2/continuous_all.csv')
rf_hyb = rf['hybrid_sharpe'].values
ca_hyb = ca['hyb_bin_sharpe'].values
diffs = np.abs(rf_hyb - ca_hyb)
print(f'hyb_bin in continuous_all.csv: {list(ca.columns)}')
# Check column names
hyb_col = [c for c in ca.columns if 'hyb' in c.lower()]
print(f'Hybrid columns in continuous_all: {hyb_col}')
print(f'RF hybrid: {rf_hyb}')
print(f'CA hybrid: {ca.columns}')
print()

# Based on column names, check the right one
for col in hyb_col:
    if 'sharpe' not in col.lower(): continue
    ca_hyb_val = ca[col].values
    diffs = np.abs(rf_hyb - ca_hyb_val)
    max_diff = diffs.max()
    print(f'  {col}: max diff={max_diff:.4f}')
    if max_diff > 0.5:
        print(f'  WARNING: Inconsistency detected in {col}!')
        errors += 1
    else:
        print(f'  OK')

# 2. Check calibration_comparison has three models
cc = pd.read_csv('btc_conditional/results_v2/calibration_comparison.csv')
print(f'\\nCalibration columns: {list(cc.columns)}')
for col in ['lstm_ece', 'xgb_ece', 'tf_ece']:
    if col in cc.columns:
        print(f'  {col}: {cc[col].mean():.4f}')
    else:
        print(f'  MISSING: {col}')
        errors += 1

# 3. Check TSLA validation exists
tsla = pd.read_csv('btc_conditional/results_v2/tsla_validation.csv')
print(f'\\nTSLA validation: {len(tsla)} folds')
tsla_hyb = tsla['hybrid_sharpe'].values
tsla_lstm = tsla['lstm_sharpe'].values
print(f'  TSLA LSTM mean: {tsla_lstm.mean():.3f}')
print(f'  TSLA Hybrid mean: {tsla_hyb.mean():.3f}')
wins = sum(tsla_hyb > tsla_lstm)
print(f'  Hybrid wins: {wins}/{len(tsla)}')

if errors > 0:
    print(f'\\nFAILED: {errors} errors')
    sys.exit(1)
else:
    print('\\nALL CHECKS PASSED')
