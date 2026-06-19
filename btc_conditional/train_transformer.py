"""Transformer baseline — 8-fold walk-forward training + evaluation"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_conditional.trainer import BTCDataLoader, BCESmoothFocalLoss, ALL_FEATS
from btc_conditional.model_transformer import TransformerModel
from btc_conditional.backtest import backtest
from btc_conditional.metrics import calc_all_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'data/processed/btc_daily_full.csv'
OUT_DIR = 'btc_conditional/runs_transformer'
RESULTS_DIR = 'btc_conditional/results_v2'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

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

THRESHOLDS = {'vol_p75': 0.093, 'crisis_ret': -0.10, 'sma_tight': 0.008, 'sma_wide': 0.025}
REGIME_THRESHOLDS = {'trend_up': 0.40, 'trend_down': 0.50, 'chop': 0.55, 'crisis': 0.80}

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

def bt_long(sigs, ret, cost=10):
    return backtest(sigs, ret, cost_bps=cost)

def train_fold(dl, tr_s, tr_e, te_s, te_e):
    dl.normalize_on_window(tr_s, tr_e)
    model = TransformerModel(n_features=len(ALL_FEATS)).to(device)
    criterion = BCESmoothFocalLoss(gamma=2.0, alpha=0.25, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    x_hist, x_real, y, ret = dl.get_window_data(tr_s, tr_e)
    if x_hist is None:
        return None, None, None, None
    
    xh = torch.tensor(x_hist, dtype=torch.float32).to(device)
    xr = torch.tensor(x_real, dtype=torch.float32).to(device)
    yt = torch.tensor(y, dtype=torch.float32).to(device)
    
    # Validation split (last 20% of training)
    n_tr = int(len(xh) * 0.8)
    xh_val, xr_val, yt_val = xh[n_tr:], xr[n_tr:], yt[n_tr:]
    xh_tr, xr_tr, yt_tr = xh[:n_tr], xr[:n_tr], yt[:n_tr]
    
    best_val_sharpe = -999
    best_state = None
    
    for epoch in range(20):
        model.train()
        # Shuffle
        perm = torch.randperm(len(xh_tr))
        loss_sum = 0
        for i in range(0, len(xh_tr), 32):
            idx = perm[i:i + 32]
            out = model(xh_tr[idx], xr_tr[idx])
            loss = criterion(out, yt_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item()
        
        # Validation
        if (epoch + 1) % 5 == 0 or epoch == 19:
            model.eval()
            with torch.no_grad():
                val_out = model(xh_val, xr_val)
            val_probs = val_out[:, 0].cpu().numpy()
            val_ret = ret[n_tr:]
            thr, m = find_best_threshold(val_probs, val_ret)
            print(f'  Epoch {epoch+1:2d}/20 | Loss {loss_sum/len(xh_tr)*32:.4f} | Val Sharpe {m["sharpe"]:+.3f} | Thr {thr:.2f}')
            if m['sharpe'] > best_val_sharpe:
                best_val_sharpe = m['sharpe']
                best_state = model.state_dict()
                best_threshold = thr
    
    # Save best model
    exp_dir = os.path.join(OUT_DIR, f'train_{tr_s[:10]}')
    os.makedirs(exp_dir, exist_ok=True)
    torch.save(best_state, os.path.join(exp_dir, 'best_model.pth'))
    with open(os.path.join(exp_dir, 'threshold.json'), 'w') as f:
        json.dump({
            'threshold': best_threshold,
            'temperature': 1.0,
            'feat_mean': dl.feat_mean.tolist(),
            'feat_std': dl.feat_std.tolist(),
        }, f)
    
    return model, best_state, best_threshold, dl.feat_mean, dl.feat_std

def find_best_threshold(probs, ret, cost_bps=10):
    if len(probs) < 5:
        return 0.5, {'sharpe': 0, 'trades': 0}
    best_sharpe = -999
    best_thr = 0.5
    for thr in np.arange(0.30, 0.81, 0.02):
        sigs = (probs >= thr).astype(int)
        bt = bt_long(sigs, ret, cost=cost_bps)
        m = calc_all_metrics(bt['net_returns'])
        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_thr = thr
    return best_thr, {'sharpe': best_sharpe, 'trades': bt['n_trades']}

def evaluate(model, dl, te_s, te_e, threshold):
    lookback = (pd.to_datetime(te_s) - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    x_hist, x_real, y, ret = dl.get_window_data(lookback, te_e, filter_training=False)
    if x_hist is None:
        return None, None, None
    model.eval()
    with torch.no_grad():
        xh_t = torch.tensor(x_hist, dtype=torch.float32).to(device)
        xr_t = torch.tensor(x_real, dtype=torch.float32).to(device)
        probs = model(xh_t, xr_t)[:, 0].cpu().numpy()
    offset = 60 - 30
    test_mask = (dl.df['datetime'] >= pd.to_datetime(te_s)) & (dl.df['datetime'] < pd.to_datetime(te_e))
    n_test = min(len(probs) - offset, test_mask.sum() - 30)
    if n_test < 5:
        return None, None, None
    probs_t = probs[offset:offset + n_test]
    ret_t = ret[offset:offset + n_test]
    return probs_t, ret_t, y[offset:offset + n_test] if y is not None else None

def main():
    print('=' * 80)
    print('TRANSFORMER BASELINE — 8-fold walk-forward')
    print('=' * 80)
    
    df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
    results = []
    
    for idx, (fname, tr_s, tr_e, te_s, te_e) in enumerate(FOLDS):
        print(f'\n{fname} [{REGIMES[idx]}] {te_s[:10]}~{te_e[:10]}')
        
        dl = BTCDataLoader(DATA_PATH, feats=ALL_FEATS, target_filter=0.005)
        model, state, thr, _, _ = train_fold(dl, tr_s, tr_e, te_s, te_e)
        if model is None:
            print('  Skip')
            continue
        
        model.load_state_dict(state)
        
        # Evaluate (pure Transformer)
        probs, ret_test, _ = evaluate(model, dl, te_s, te_e, thr)
        if probs is None:
            print('  No predictions')
            continue
        
        sigs = (probs >= thr).astype(int)
        bt = bt_long(sigs, ret_test)
        m = calc_all_metrics(bt['net_returns'])
        m['trades'] = bt['n_trades']
        
        # Transformer + Regime
        df_te = df[(df['datetime'] >= pd.to_datetime(te_s)) & (df['datetime'] < pd.to_datetime(te_e))].copy()
        reg_sigs = []
        for i in range(len(probs)):
            r = regime(df_te.iloc[i + 30] if i + 30 < len(df_te) else df_te.iloc[-1])
            thr_r = REGIME_THRESHOLDS.get(r, 0.50)
            reg_sigs.append(1 if probs[i] >= thr_r else 0)
        reg_sigs = np.array(reg_sigs)
        bt_h = bt_long(reg_sigs, ret_test)
        m_h = calc_all_metrics(bt_h['net_returns'])
        m_h['trades'] = bt_h['n_trades']
        
        print(f'  Transformer:  Sharpe={m["sharpe"]:+.3f}  Trades={m["trades"]}')
        print(f'  TF + Regime:  Sharpe={m_h["sharpe"]:+.3f}  Trades={m_h["trades"]}')
        
        results.append({
            'fold': fname, 'regime': REGIMES[idx],
            'tf_sharpe': m['sharpe'], 'tf_trades': m['trades'],
            'tf_hybrid_sharpe': m_h['sharpe'], 'tf_hybrid_trades': m_h['trades'],
        })
    
    # Summary
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f"{'Fold':<8} {'Regime':>6} | {'TF_SR':>7} {'TF+T':>4} {'TF+Regime':>10} {'TF+Reg_T':>7}")
    print('-' * 55)
    for r in results:
        print(f"{r['fold']:<8} {r['regime']:>6} | {r['tf_sharpe']:>+7.3f} {r['tf_trades']:>4d} {r['tf_hybrid_sharpe']:>+10.3f} {r['tf_hybrid_trades']:>4d}")
    print('-' * 55)
    avg_tf = np.mean([r['tf_sharpe'] for r in results])
    avg_tfh = np.mean([r['tf_hybrid_sharpe'] for r in results])
    print(f"{'Average':<8} {'':>6} | {avg_tf:>+7.3f} {'':>4} {avg_tfh:>+10.3f}")
    
    pd.DataFrame(results).to_csv(f'{RESULTS_DIR}/baseline_transformer.csv', index=False)
    print(f'\nSaved to {RESULTS_DIR}/baseline_transformer.csv')

if __name__ == '__main__':
    main()
