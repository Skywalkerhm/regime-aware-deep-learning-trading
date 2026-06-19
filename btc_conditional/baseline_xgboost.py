"""XGBoost baseline — walk-forward evaluation + regime hybrid variant"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from btc_conditional.backtest import backtest
from btc_conditional.metrics import calc_all_metrics
from btc_conditional.trainer import BTCDataLoader, ALL_FEATS

DATA_PATH = 'data/processed/btc_daily_full.csv'
RESULTS_DIR = 'btc_conditional/results_v2'
os.makedirs(RESULTS_DIR, exist_ok=True)

THRESHOLDS = {'vol_p75': 0.093, 'crisis_ret': -0.10, 'sma_tight': 0.008, 'sma_wide': 0.025}

def regime(row):
    v = row['volatility_20d']
    sr = row['sma_ratio_20_50']
    r20 = row['return_20d']
    if v > THRESHOLDS['vol_p75'] * 2 or r20 < THRESHOLDS['crisis_ret']:
        return 'crisis'
    if abs(sr - 1) > THRESHOLDS['sma_wide']:
        return 'trend_up' if sr > 1 else 'trend_down'
    if abs(sr - 1) > THRESHOLDS['sma_tight']:
        return 'trend_up' if sr > 1 else 'trend_down'
    return 'chop'

REGIME_THRESHOLDS = {'trend_up': 0.40, 'trend_down': 0.50, 'chop': 0.55, 'crisis': 0.80}

def prep_df(df):
    close = df['price_close']
    df['sma20'] = close.rolling(20).mean()
    df['sma50'] = close.rolling(50).mean()
    delta = close.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi14'] = 100 - 100 / (1 + g / (l + 1e-10))
    return df

def bt_long(sigs, ret, cost=10):
    return backtest(sigs, ret, cost_bps=cost)

FOLDS = [
    ('Fold 1', '2018-07-01', '2020-07-01', '2020-07-01', '2021-01-01'),
    ('Fold 2', '2019-07-01', '2021-07-01', '2021-07-01', '2022-01-01'),
    ('Fold 3', '2020-07-01', '2022-07-01', '2022-07-01', '2023-01-01'),
    ('Fold 4', '2021-07-01', '2023-07-01', '2023-07-01', '2024-01-01'),
    ('Fold 5', '2022-01-01', '2024-01-01', '2024-01-01', '2024-07-01'),
    ('Fold 6', '2022-07-01', '2024-07-01', '2024-07-01', '2025-01-01'),
    ('Fold 7', '2023-01-01', '2025-01-01', '2025-01-01', '2025-07-01'),
    ('Fold 8', '2023-07-01', '2025-07-01', '2025-07-01', '2026-01-01'),
]
REGIMES = ['Recov', 'Peak', 'Crash', 'Recov', 'Bull', 'Bull', 'Bull', 'Bear']

def prepare_fold_data(df, tr_s, tr_e, te_s, te_e):
    """Split fold into train/test, align features/targets."""
    train_mask = (df['datetime'] >= pd.to_datetime(tr_s)) & (df['datetime'] < pd.to_datetime(tr_e))
    test_mask = (df['datetime'] >= pd.to_datetime(te_s)) & (df['datetime'] < pd.to_datetime(te_e))
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    # Features: ALL_FEATS (34 historical features, no x_real)
    X_train = df_train[ALL_FEATS].values.astype(np.float32)
    X_test = df_test[ALL_FEATS].values.astype(np.float32)
    
    # Target: next-day direction (shift returns forward)
    y_train = (df_train['return_1d'].shift(-1).fillna(0) > 0).astype(int).values
    y_test = (df_test['return_1d'].shift(-1).fillna(0) > 0).astype(int).values
    
    # Test returns (for backtest)
    ret_test = df_test['return_1d'].values
    
    return X_train, y_train, X_test, y_test, ret_test, df_test

def main():
    print('=' * 80)
    print('XGBoost BASELINE — 8-fold walk-forward evaluation')
    print('=' * 80)
    
    df_full = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
    df_full = prep_df(df_full)
    print(f'Data: {len(df_full)} rows, {len(ALL_FEATS)} features\n')
    
    results = []
    
    for idx, (fname, tr_s, tr_e, te_s, te_e) in enumerate(FOLDS):
        reg_label = REGIMES[idx]
        print(f'{fname} [{reg_label}] {te_s[:10]}~{te_e[:10]}')
        
        X_tr, y_tr, X_te, y_te, ret_te, df_te = prepare_fold_data(df_full, tr_s, tr_e, te_s, te_e)
        
        if len(X_tr) < 100 or len(X_te) < 20:
            print(f'  Skip: insufficient data ({len(X_tr)} train, {len(X_te)} test)')
            continue
        
        # Handle NaN (first few rows in rolling features)
        mask_nan_tr = ~np.isnan(X_tr).any(axis=1)
        mask_nan_te = ~np.isnan(X_te).any(axis=1)
        X_tr, y_tr = X_tr[mask_nan_tr], y_tr[mask_nan_tr]
        X_te, y_te, ret_te = X_te[mask_nan_te], y_te[mask_nan_te], ret_te[mask_nan_te]
        
        if len(X_tr) < 50 or len(X_te) < 10:
            print(f'  Skip: too many NaNs ({len(X_tr)} train, {len(X_te)} test)')
            continue
        
        # ── Train XGBoost ──
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic', eval_metric='logloss',
            random_state=42, n_jobs=1
        )
        model.fit(X_tr, y_tr)
        
        # ── Predict probabilities ──
        probs = model.predict_proba(X_te)[:, 1]
        
        # ── Strategy 1: XGBoost (pure, fixed threshold) ──
        val_split = int(len(X_tr) * 0.8)
        # Find best threshold on validation portion of training
        X_val, y_val = X_tr[val_split:], y_tr[val_split:]
        if len(X_val) > 10:
            val_probs = model.predict_proba(X_val)[:, 1]
            _, val_ret = np.ones(len(val_probs)), np.zeros(len(val_probs))  # placeholder
            best_thr = 0.5
        else:
            best_thr = 0.5
        
        xgb_sigs = (probs >= best_thr).astype(int)
        xgb_bt = bt_long(xgb_sigs, ret_te)
        xgb_m = calc_all_metrics(xgb_bt['net_returns'])
        xgb_m['trades'] = xgb_bt['n_trades']
        
        # ── Strategy 2: XGBoost + Regime (hybrid, state-dependent thresholds) ──
        hyb_sigs = []
        test_start = df_te[mask_nan_te].iloc[0].name
        for i in range(len(probs)):
            row = df_te.iloc[i]
            r = regime(row)
            thr = REGIME_THRESHOLDS.get(r, 0.50)
            hyb_sigs.append(1 if probs[i] >= thr else 0)
        hyb_sigs = np.array(hyb_sigs)
        
        hyb_bt = bt_long(hyb_sigs, ret_te)
        hyb_m = calc_all_metrics(hyb_bt['net_returns'])
        hyb_m['trades'] = hyb_bt['n_trades']
        
        # ── BH baseline on same test period ──
        bh_m = calc_all_metrics(ret_te)
        
        print(f'  XGBoost:   Sharpe={xgb_m["sharpe"]:+.3f}  Trades={xgb_m["trades"]}')
        print(f'  XGB+Regime: Sharpe={hyb_m["sharpe"]:+.3f}  Trades={hyb_m["trades"]}')
        print(f'  Buy&Hold:  Sharpe={bh_m["sharpe"]:+.3f}')
        print()
        
        results.append({
            'fold': fname, 'regime': reg_label,
            'xgb_sharpe': xgb_m['sharpe'], 'xgb_trades': xgb_m['trades'],
            'xgb_hybrid_sharpe': hyb_m['sharpe'], 'xgb_hybrid_trades': hyb_m['trades'],
            'bh_sharpe': bh_m['sharpe'],
        })
    
    # ═══ Summary ═══
    print('=' * 80)
    print('SUMMARY TABLE')
    print('=' * 80)
    hdr = f"{'Fold':<8} {'Regime':>6} | {'XGB_SR':>7} {'XGB+T':>4} {'XGB+Regime':>10} {'XGB+Reg_T':>7} {'BH_SR':>7}"
    print(hdr)
    print('-' * 60)
    for r in results:
        print(f"{r['fold']:<8} {r['regime']:>6} | {r['xgb_sharpe']:>+7.3f} {r['xgb_trades']:>4d} {r['xgb_hybrid_sharpe']:>+10.3f} {r['xgb_hybrid_trades']:>4d} {r['bh_sharpe']:>+7.3f}")
    
    print('-' * 60)
    avg_xgb = np.mean([r['xgb_sharpe'] for r in results])
    avg_xgbh = np.mean([r['xgb_hybrid_sharpe'] for r in results])
    avg_bh = np.mean([r['bh_sharpe'] for r in results])
    print(f"{'Average':<8} {'':>6} | {avg_xgb:>+7.3f} {'':>4} {avg_xgbh:>+10.3f} {'':>4} {avg_bh:>+7.3f}")
    
    # Save
    pd.DataFrame(results).to_csv(f'{RESULTS_DIR}/baseline_xgboost.csv', index=False)
    print(f'\nSaved to {RESULTS_DIR}/baseline_xgboost.csv')

if __name__ == '__main__':
    main()
