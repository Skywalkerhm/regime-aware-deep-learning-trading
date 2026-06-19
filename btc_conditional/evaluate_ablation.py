"""消融实验评估：排除 today_Open_pct 后的 Hybrid vs LSTM

x_real 置空，模型只能依赖历史窗口特征（全部来自 t-1 及更早）。
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_conditional.backtest import backtest
from btc_conditional.metrics import calc_all_metrics
from btc_conditional.trainer import BTCTrainer, BTCDataLoader, ALL_FEATS
from scipy import stats

DATA_PATH = "data/processed/btc_daily_full.csv"
RESULTS_DIR = "btc_conditional/results_v2"

THRESHOLDS = {'vol_p75': 0.093, 'crisis_ret': -0.10, 'sma_tight': 0.008, 'sma_wide': 0.025}

def regime(row):
    v = row['volatility_20d']; sr = row['sma_ratio_20_50']; r20 = row['return_20d']
    if v > THRESHOLDS['vol_p75']*2 or r20 < THRESHOLDS['crisis_ret']: return 'crisis'
    if abs(sr-1) > THRESHOLDS['sma_wide']: return 'trend_up' if sr > 1 else 'trend_down'
    if abs(sr-1) > THRESHOLDS['sma_tight']: return 'trend_up' if sr > 1 else 'trend_down'
    return 'chop'

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

def load_val_threshold(ckpt_path):
    threshold_path = ckpt_path.replace("best_model.pth", "threshold.json")
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            data = json.load(f)
            return (data["threshold"], data.get("temperature", 1.0),
                    data.get("feat_mean"), data.get("feat_std"))
    return None, None, None, None

FOLDS = [
    ("Fold 1", "2018-07-01", "2020-07-01", "2020-07-01", "2021-01-01", "Recov"),
    ("Fold 2", "2019-07-01", "2021-07-01", "2021-07-01", "2022-01-01", "Peak"),
    ("Fold 3", "2020-07-01", "2022-07-01", "2022-07-01", "2023-01-01", "Crash"),
    ("Fold 4", "2021-07-01", "2023-07-01", "2023-07-01", "2024-01-01", "Recov"),
    ("Fold 5", "2022-01-01", "2024-01-01", "2024-01-01", "2024-07-01", "Bull"),
    ("Fold 6", "2022-07-01", "2024-07-01", "2024-07-01", "2025-01-01", "Bull"),
    ("Fold 7", "2023-01-01", "2025-01-01", "2025-01-01", "2025-07-01", "Bull"),
    ("Fold 8", "2023-07-01", "2025-07-01", "2025-07-01", "2026-01-01", "Bear"),
]

def main():
    print("=" * 80)
    print("ABLATION EVALUATION: 排除 today_Open_pct (x_real 置空)")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
    df = prep_df(df)

    config = {"window": {"seq_len": 30}, "model": {"dropout": 0.3}, "trainer": {"lr": 3e-4, "epochs": 15}}
    results = []

    for fname, tr_s, tr_e, te_s, te_e, reg_label in FOLDS:
        print(f"\n{fname} [{reg_label}] {te_s[:10]}~{te_e[:10]}")

        bp = f'btc_conditional/runs_ablation/train_{tr_s[:10]}/best_model.pth'
        if not os.path.exists(bp):
            print(f"  No checkpoint: {bp}")
            continue

        val_thr, _, feat_mean, feat_std = load_val_threshold(bp)
        if val_thr is None:
            print("  No threshold")
            continue

        # n_real=0, real_feats=[] — 与训练一致
        t = BTCTrainer(config, feats=ALL_FEATS, n_real=0)
        t.model.load_state_dict(torch.load(bp, map_location='cpu'))
        dl = BTCDataLoader(DATA_PATH, feats=ALL_FEATS, real_feats=[])
        if feat_mean is not None and feat_std is not None:
            dl.apply_saved_norm(feat_mean, feat_std)

        msk = (df.datetime >= pd.to_datetime(te_s)) & (df.datetime < pd.to_datetime(te_e))
        tr = df.loc[msk, 'return_1d'].values

        # B&H
        bh_m = calc_all_metrics(tr)

        # SMA (SMA20/50 crossover)
        close, s20, s50 = df['price_close'], df['sma20'], df['sma50']
        sig_sma = (s20 > s50).astype(int).loc[msk].values
        sma_m = calc_all_metrics(bt_long(sig_sma, tr)['net_returns'])

        # LSTM
        lb = (pd.to_datetime(te_s) - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
        probs, y, ret = t.predict(dl, lb, te_e)
        if probs is None:
            print("  No predictions")
            continue
        off = 60 - 30
        n = min(len(probs) - off, max(0, msk.sum() - 30))
        if n < 5:
            print("  <5 test samples")
            continue
        pt = probs[off:off+n]
        rt = ret[off:off+n]

        sigs_lstm = (pt >= val_thr).astype(int)
        bt_lstm = bt_long(sigs_lstm, rt)
        lm = calc_all_metrics(bt_lstm['net_returns'])

        # Hybrid
        test_idx = df[msk].index[30:30+n]
        hyb_signals = []
        for i, idx in enumerate(test_idx):
            r = regime(df.loc[idx])
            thr = {'trend_up': 0.40, 'trend_down': 0.50, 'chop': 0.55, 'crisis': 0.80}.get(r, 0.5)
            hyb_signals.append(1 if pt[i] >= thr else 0)
        bt_hyb = bt_long(np.array(hyb_signals), rt)
        hm = calc_all_metrics(bt_hyb['net_returns'])

        improvement = hm['sharpe'] - lm['sharpe']
        print(f"  LSTM={lm['sharpe']:+.3f} Hybrid={hm['sharpe']:+.3f} Δ={improvement:+.3f} | SMA={sma_m['sharpe']:+.3f} BH={bh_m['sharpe']:+.3f}")

        results.append({
            'fold': fname, 'regime': reg_label,
            'bh_sharpe': bh_m['sharpe'],
            'sma_sharpe': sma_m['sharpe'],
            'lstm_sharpe': lm['sharpe'],
            'hybrid_sharpe': hm['sharpe'],
            'improvement': improvement,
        })

    # Summary
    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY (no today_Open_pct)")
    print("=" * 80)
    hdr = f"{'Fold':<8} {'Regime':>7} {'BH':>8} {'SMA':>8} {'LSTM':>8} {'Hybrid':>8} {'Delta':>8}"
    print(hdr + "\n" + "-" * 60)
    for r in results:
        print(f"{r['fold']:<8} {r['regime']:>7} {r['bh_sharpe']:>+8.3f} {r['sma_sharpe']:>+8.3f} "
              f"{r['lstm_sharpe']:>+8.3f} {r['hybrid_sharpe']:>+8.3f} {r['improvement']:>+8.3f}")

    lstm_arr = np.array([r['lstm_sharpe'] for r in results])
    hyb_arr = np.array([r['hybrid_sharpe'] for r in results])
    imp_arr = hyb_arr - lstm_arr

    print(f"\n{'Average':<8} {'':>7} {np.mean([r['bh_sharpe'] for r in results]):>+8.3f} "
          f"{np.mean([r['sma_sharpe'] for r in results]):>+8.3f} "
          f"{np.mean(lstm_arr):>+8.3f} {np.mean(hyb_arr):>+8.3f} {np.mean(imp_arr):>+8.3f}")
    print(f"{'Std (ddof=1)':<8} {'':>7} {np.std([r['bh_sharpe'] for r in results], ddof=1):>8.3f} "
          f"{np.std([r['sma_sharpe'] for r in results], ddof=1):>8.3f} "
          f"{np.std(lstm_arr, ddof=1):>8.3f} {np.std(hyb_arr, ddof=1):>8.3f}")

    t_stat, p_val = stats.ttest_rel(hyb_arr, lstm_arr)
    print(f"\nPaired t-test: p={p_val:.4f}")
    try:
        w_stat, w_pval = stats.wilcoxon(hyb_arr, lstm_arr)
        print(f"Wilcoxon: p={w_pval:.4f}")
    except:
        pass

    wins = sum(1 for r in results if r['improvement'] > 0)
    print(f"Hybrid wins: {wins}/{len(results)}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/ablation_no_open_pct.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/ablation_no_open_pct.csv")

if __name__ == "__main__":
    main()
