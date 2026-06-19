"""HMM regime detector — integrate with LSTM Hybrid strategy"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_conditional.backtest import backtest
from btc_conditional.metrics import calc_all_metrics
from btc_conditional.trainer import BTCTrainer, BTCDataLoader, ALL_FEATS
from btc_conditional.regime_hmm import HMMRegimeDetector

DATA_PATH = 'data/processed/btc_daily_full.csv'
RESULTS_DIR = 'btc_conditional/results_v2'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Rule-based thresholds (for comparison and crisis override)
RULE_THRESHOLDS = {'vol_p75': 0.093, 'crisis_ret': -0.10, 'sma_tight': 0.008, 'sma_wide': 0.025}
REGIME_THRESHOLDS = {'trend_up': 0.40, 'trend_down': 0.50, 'chop': 0.55, 'crisis': 0.80}

def rule_regime(row):
    v = row['volatility_20d']; sr = row['sma_ratio_20_50']; r20 = row['return_20d']
    if v > RULE_THRESHOLDS['vol_p75'] * 2 or r20 < RULE_THRESHOLDS['crisis_ret']:
        return 'crisis'
    if abs(sr - 1) > RULE_THRESHOLDS['sma_wide']:
        return 'trend_up' if sr > 1 else 'trend_down'
    if abs(sr - 1) > RULE_THRESHOLDS['sma_tight']:
        return 'trend_up' if sr > 1 else 'trend_down'
    return 'chop'

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

def bt_long(sigs, ret, cost=10):
    return backtest(sigs, ret, cost_bps=cost)

def load_val_threshold(ckpt_path):
    thr_path = ckpt_path.replace('best_model.pth', 'threshold.json')
    if os.path.exists(thr_path):
        with open(thr_path) as f:
            data = json.load(f)
        return data['threshold'], data.get('temperature', 1.0), data.get('feat_mean'), data.get('feat_std')
    return None, None, None, None

def prep_df(df):
    close = df['price_close']
    df['sma20'] = close.rolling(20).mean()
    df['sma50'] = close.rolling(50).mean()
    delta = close.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi14'] = 100 - 100 / (1 + g / (l + 1e-10))
    return df

def main():
    print('=' * 80)
    print('HMM REGIME — compare HMM-based vs Rule-based regime gating')
    print('=' * 80)
    
    df_full = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
    df_full = prep_df(df_full)
    
    rule_results = []
    hmm_results = []
    
    for idx, (fname, tr_s, tr_e, te_s, te_e) in enumerate(FOLDS):
        reg_label = REGIMES[idx]
        print(f'\n{fname} [{reg_label}] {te_s[:10]}~{te_e[:10]}')
        
        # Load LSTM model
        bp = next((p for p in [
            f'btc_conditional/runs_v7/train_{tr_s[:10]}/best_model.pth',
        ] if os.path.exists(p)), None)
        
        if not bp:
            print('  No checkpoint')
            continue
        
        trainer = BTCTrainer({"window": {"seq_len": 30}, "model": {"dropout": 0.3}, "trainer": {"lr": 3e-4, "epochs": 15}}, feats=ALL_FEATS)
        trainer.model.load_state_dict(torch.load(bp, map_location='cpu'))
        dl = BTCDataLoader(DATA_PATH, feats=ALL_FEATS)
        
        val_thr, _, feat_mean, feat_std = load_val_threshold(bp)
        if val_thr is None:
            print('  No threshold')
            continue
        if feat_mean is not None:
            dl.apply_saved_norm(feat_mean, feat_std)
        
        # Predict LSTM probabilities
        lb = (pd.to_datetime(te_s) - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
        probs, y, ret = trainer.predict(dl, lb, te_e)
        if probs is None:
            continue
        
        offset = 60 - 30
        test_mask = (df_full['datetime'] >= pd.to_datetime(te_s)) & (df_full['datetime'] < pd.to_datetime(te_e))
        n_test = min(len(probs) - offset, test_mask.sum() - 30)
        if n_test < 5:
            continue
        
        probs_t = probs[offset:offset + n_test]
        ret_t = ret[offset:offset + n_test]
        test_idx = df_full[test_mask].index[30:30 + n_test]
        df_te = df_full.loc[test_idx]
        
        # ── Rule-based regime (baseline) ──
        rule_signals = []
        for i in range(len(probs_t)):
            r = rule_regime(df_te.iloc[i])
            thr = REGIME_THRESHOLDS.get(r, 0.50)
            rule_signals.append(1 if probs_t[i] >= thr else 0)
        rule_sigs = np.array(rule_signals)
        rule_bt = bt_long(rule_sigs, ret_t)
        rule_m = calc_all_metrics(rule_bt['net_returns'])
        rule_m['trades'] = rule_bt['n_trades']
        
        # ── HMM-based regime ──
        # Fit HMM on training data only
        hmm_detector = HMMRegimeDetector(n_states=3)
        train_df = df_full[(df_full['datetime'] >= pd.to_datetime(tr_s)) & (df_full['datetime'] < pd.to_datetime(tr_e))]
        hmm_detector.fit(train_df)
        
        hmm_signals = []
        for i in range(len(probs_t)):
            row = df_te.iloc[i]
            # Crisis override: HMM can miss rare crisis events
            v = row['volatility_20d']
            r20 = row['return_20d']
            if v > RULE_THRESHOLDS['vol_p75'] * 2 or r20 < RULE_THRESHOLDS['crisis_ret']:
                r = 'crisis'
            else:
                r = hmm_detector.predict(df_te, i)
            thr = REGIME_THRESHOLDS.get(r, 0.50)
            hmm_signals.append(1 if probs_t[i] >= thr else 0)
        hmm_sigs = np.array(hmm_signals)
        hmm_bt = bt_long(hmm_sigs, ret_t)
        hmm_m = calc_all_metrics(hmm_bt['net_returns'])
        hmm_m['trades'] = hmm_bt['n_trades']
        
        hmm_improvement = hmm_m['sharpe'] - rule_m['sharpe']
        
        print(f'  Rule-based: Sharpe={rule_m["sharpe"]:+.3f}  Trades={rule_m["trades"]}')
        print(f'  HMM-based:  Sharpe={hmm_m["sharpe"]:+.3f}  Trades={hmm_m["trades"]}')
        print(f'  \u0394(HMM-Rule): {hmm_improvement:+.3f}')
        
        rule_results.append({
            'fold': fname, 'regime': reg_label,
            'sharpe': rule_m['sharpe'], 'trades': rule_m['trades'],
        })
        hmm_results.append({
            'fold': fname, 'regime': reg_label,
            'sharpe': hmm_m['sharpe'], 'trades': hmm_m['trades'],
        })
    
    # ── Summary ──
    print('\n' + '=' * 80)
    print('SUMMARY: Rule-based vs HMM-based regime')
    print('=' * 80)
    print(f"{'Fold':<8} {'Regime':>6} | {'Rule_SR':>8} {'HMM_SR':>8} {'\u0394(H-R)':>8}")
    print('-' * 45)
    for r, h in zip(rule_results, hmm_results):
        d = h['sharpe'] - r['sharpe']
        print(f"{r['fold']:<8} {r['regime']:>6} | {r['sharpe']:>+8.3f} {h['sharpe']:>+8.3f} {d:>+8.3f}")
    
    print('-' * 45)
    avg_r = np.mean([r['sharpe'] for r in rule_results])
    avg_h = np.mean([r['sharpe'] for r in hmm_results])
    print(f"{'Average':<8} {'':>6} | {avg_r:>+8.3f} {avg_h:>+8.3f} {avg_h - avg_r:>+8.3f}")
    
    # ── Statistical test ──
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(
        [h['sharpe'] for h in hmm_results],
        [r['sharpe'] for r in rule_results]
    )
    print(f'\nPaired t-test (HMM vs Rule): p={p_val:.4f}')
    wins = sum(1 for r, h in zip(rule_results, hmm_results) if h['sharpe'] > r['sharpe'])
    print(f'HMM wins: {wins}/{len(rule_results)} folds')
    
    # Save
    pd.DataFrame({
        'fold': [r['fold'] for r in rule_results],
        'regime': [r['regime'] for r in rule_results],
        'rule_sharpe': [r['sharpe'] for r in rule_results],
        'rule_trades': [r['trades'] for r in rule_results],
        'hmm_sharpe': [h['sharpe'] for h in hmm_results],
        'hmm_trades': [h['trades'] for h in hmm_results],
    }).to_csv(f'{RESULTS_DIR}/regime_hmm.csv', index=False)
    print(f'\nSaved to {RESULTS_DIR}/regime_hmm.csv')

if __name__ == '__main__':
    main()
