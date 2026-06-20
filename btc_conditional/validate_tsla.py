import os, sys, json
import numpy as np, pandas as pd, torch
import warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btc_conditional.backtest import backtest
from btc_conditional.metrics import calc_all_metrics
from btc_conditional.trainer import BTCTrainer, BTCDataLoader, ALL_FEATS

DATA_PATH = 'data/processed/tsla_full.csv'
RESULTS_DIR = 'btc_conditional/results_v2'
os.makedirs(RESULTS_DIR, exist_ok=True)

THRESHOLDS = {'vol_p75': 0.093, 'crisis_ret': -0.10, 'sma_tight': 0.008}
REGIME_THR = {'trend_up': 0.40, 'trend_down': 0.50, 'chop': 0.55, 'crisis': 0.80}

def regime(row):
    v = row['volatility_20d']; sr = row['sma_ratio_20_50']; r20 = row['return_20d']
    if v > THRESHOLDS['vol_p75']*2 or r20 < THRESHOLDS['crisis_ret']:
        return 'crisis'
    if abs(sr-1) > THRESHOLDS['sma_tight']:
        return 'trend_up' if sr > 1 else 'trend_down'
    return 'chop'

def bt_long(s, r, c=10): return backtest(s, r, cost_bps=c)

FOLDS = [
    ('TF 1','2018-07-01','2020-07-01','2020-07-01','2021-01-01','Recov'),
    ('TF 2','2019-07-01','2021-07-01','2021-07-01','2022-01-01','Peak'),
    ('TF 3','2020-07-01','2022-07-01','2022-07-01','2023-01-01','Crash'),
    ('TF 4','2021-07-01','2023-07-01','2023-07-01','2024-01-01','Recov'),
    ('TF 5','2022-01-01','2024-01-01','2024-01-01','2024-07-01','Bull'),
    ('TF 6','2022-07-01','2024-07-01','2024-07-01','2025-01-01','Bull'),
    ('TF 7','2023-01-01','2025-01-01','2025-01-01','2025-07-01','Bull'),
    ('TF 8','2023-07-01','2025-07-01','2025-07-01','2026-01-01','Bear'),
]
config = {'window': {'seq_len': 30}, 'model': {'dropout': 0.3}, 'trainer': {'lr': 3e-4, 'epochs': 15}}

def prep_df(df):
    close = df['price_close']
    df['sma20'] = close.rolling(20).mean()
    df['sma50'] = close.rolling(50).mean()
    d = close.diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    df['rsi14'] = 100 - 100 / (1 + g / (l + 1e-10))
    return df

def load_val_threshold(ckpt_path):
    thr_path = ckpt_path.replace('best_model.pth', 'threshold.json')
    if os.path.exists(thr_path):
        with open(thr_path) as f:
            d = json.load(f)
        return d['threshold'], d.get('temperature', 1.0), d.get('feat_mean'), d.get('feat_std')
    return None, None, None, None

print('TSLA HYBRID VALIDATION')
print('=' * 70)

df_full = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
df_full = prep_df(df_full)
results = []

for fname, tr_s, tr_e, te_s, te_e, reg_label in FOLDS:
    bp = f'btc_conditional/runs_tsla/train_{tr_s[:10]}/best_model.pth'
    if not os.path.exists(bp):
        print(f'{fname}: no model')
        continue
    
    trainer = BTCTrainer(config, feats=ALL_FEATS)
    trainer.model.load_state_dict(torch.load(bp, map_location='cpu'))
    dl = BTCDataLoader(DATA_PATH, feats=ALL_FEATS)
    val_thr, _, feat_mean, feat_std = load_val_threshold(bp)
    if feat_mean is not None:
        dl.apply_saved_norm(feat_mean, feat_std)
    if val_thr is None:
        print(f'{fname}: no threshold')
        continue
    
    lb = (pd.to_datetime(te_s) - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    probs, y, ret = trainer.predict(dl, lb, te_e)
    if probs is None:
        continue
    
    offset = 60 - 30
    msk = (df_full['datetime'] >= pd.to_datetime(te_s)) & (df_full['datetime'] < pd.to_datetime(te_e))
    n = min(len(probs) - offset, msk.sum() - 30)
    if n < 5:
        continue
    
    pt = probs[offset:offset+n]
    rt = ret[offset:offset+n]
    te_idx = df_full[msk].index[30:30+n]
    df_te = df_full.loc[te_idx]
    
    # BH
    bh_m = calc_all_metrics(rt)
    
    # LSTM (pure)
    lstm_sigs = (pt >= val_thr).astype(int)
    lstm_bt = bt_long(lstm_sigs, rt)
    lstm_m = calc_all_metrics(lstm_bt['net_returns'])
    lstm_m['trades'] = lstm_bt['n_trades']
    
    # Hybrid (LSTM + regime)
    hyb_sigs = np.array([1 if pt[i] >= REGIME_THR.get(regime(df_te.iloc[i]), 0.50) else 0 for i in range(len(pt))])
    hyb_bt = bt_long(hyb_sigs, rt)
    hyb_m = calc_all_metrics(hyb_bt['net_returns'])
    hyb_m['trades'] = hyb_bt['n_trades']
    
    imp = hyb_m['sharpe'] - lstm_m['sharpe']
    print(f'  {fname} [{reg_label:>5}]: LSTM={lstm_m["sharpe"]:+.3f}({lstm_m["trades"]}t)  Hybrid={hyb_m["sharpe"]:+.3f}({hyb_m["trades"]}t)  BH={bh_m["sharpe"]:+.3f}  \u0394={imp:+.3f}')
    
    results.append({'fold':fname,'regime':reg_label,'bh_sharpe':bh_m['sharpe'],
        'lstm_sharpe':lstm_m['sharpe'],'lstm_trades':lstm_m['trades'],
        'hybrid_sharpe':hyb_m['sharpe'],'hybrid_trades':hyb_m['trades'],'improvement':imp})

# Summary
print('\n' + '=' * 70)
print(f'{"Fold":<6} {"Regime":>6} | {"BH":>7} {"LSTM":>7} {"T":>3} {"Hybrid":>8} {"T":>3} | {"\u0394":>7}')
print('-' * 55)
for r in results:
    print(f'{r["fold"]:<6} {r["regime"]:>6} | {r["bh_sharpe"]:>+7.3f} {r["lstm_sharpe"]:>+7.3f} {r["lstm_trades"]:>3d} {r["hybrid_sharpe"]:>+8.3f} {r["hybrid_trades"]:>3d} | {r["improvement"]:>+7.3f}')
print('-' * 55)
avg_lstm = np.mean([r['lstm_sharpe'] for r in results])
avg_hyb = np.mean([r['hybrid_sharpe'] for r in results])
avg_bh = np.mean([r['bh_sharpe'] for r in results])
wins = sum(1 for r in results if r['improvement'] > 0)
print(f'{"Avg":<6} {"":>6} | {avg_bh:>+7.3f} {avg_lstm:>+7.3f} {"":>3} {avg_hyb:>+8.3f} {"":>3} | {avg_hyb-avg_lstm:>+7.3f}')
print(f'\nHybrid wins: {wins}/{len(results)} folds')

# Statistical test
from scipy import stats
lstm_s = [r['lstm_sharpe'] for r in results]
hyb_s = [r['hybrid_sharpe'] for r in results]
t_stat, p_val = stats.ttest_rel(hyb_s, lstm_s)
print(f'Paired t-test: p={p_val:.4f}')
w_stat, w_pval = stats.wilcoxon(hyb_s, lstm_s)
print(f'Wilcoxon: p={w_pval:.4f}')

# Save
import csv as csv_module
with open(f'{RESULTS_DIR}/tsla_validation.csv', 'w', newline='') as f:
    w = csv_module.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
print(f'\nSaved to {RESULTS_DIR}/tsla_validation.csv')
