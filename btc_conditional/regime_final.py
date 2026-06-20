"""市场状态感知策略 Final — 三路对比：Regime vs LSTM vs LSTM+Regime"""
import os, sys, json
import numpy as np, pandas as pd, torch
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btc_conditional.backtest import backtest
from btc_conditional.metrics import calc_all_metrics
from btc_conditional.trainer import BTCTrainer, BTCDataLoader, ALL_FEATS

DATA_PATH = "data/processed/btc_daily_full.csv"
RESULTS_DIR = "btc_conditional/results_v2"

THRESHOLDS = {'vol_p75': 0.093, 'crisis_ret': -0.10, 'sma_tight': 0.008, 'sma_wide': 0.025}
def load_val_threshold(ckpt_path):
    """加载验证集上选定的阈值、温度和归一化统计量"""
    threshold_path = ckpt_path.replace("best_model.pth", "threshold.json")
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            data = json.load(f)
            return (data["threshold"], data.get("temperature", 1.0),
                    data.get("feat_mean"), data.get("feat_std"))
    return None, None, None, None

# ── 状态检测 (v1 固定阈值，已验证OK) ──

def regime(row):
    v = row['volatility_20d']; sr = row['sma_ratio_20_50']; r20 = row['return_20d']
    if v > THRESHOLDS['vol_p75']*2 or r20 < THRESHOLDS['crisis_ret']: return 'crisis'
    if abs(sr-1) > THRESHOLDS['sma_tight']: return 'trend_up' if sr > 1 else 'trend_down'
    return 'chop'

def prep_df(df):
    """
    预处理：计算技术指标
    
    SMA/RSI 是滚动指标，t 时刻只用过去值，不存在未来泄漏。
    直接对全量数据用 rolling 计算是安全的。
    """
    close = df['price_close']
    df['sma20'] = close.rolling(20).mean()
    df['sma50'] = close.rolling(50).mean()
    delta = close.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi14'] = 100 - 100 / (1 + g / (l + 1e-10))
    return df

# ── 策略 ──
def bt_long(sigs, ret, cost=10): return backtest(sigs, ret, cost_bps=cost)

def strat_regime(df, te_s, te_e, cost=10):
    m = (df.datetime>=pd.to_datetime(te_s))&(df.datetime<pd.to_datetime(te_e))
    idx = df[m].index; sigs = []; prev = 0
    for i in idx:
        r = regime(df.loc[i])
        if r == 'trend_up': s = 1 if df.loc[i,'sma20']>df.loc[i,'sma50'] else 0
        elif r == 'trend_down': s = 0
        elif r == 'chop':
            rsi = df.loc[i,'rsi14']; rsi = 50 if pd.isna(rsi) else rsi
            s = 1 if rsi < 35 else (0 if rsi > 65 else prev)
        else: s = 0
        sigs.append(s); prev = s
    bt = bt_long(np.array(sigs), df.loc[idx,'return_1d'].values, cost)
    m = calc_all_metrics(bt['net_returns']); m['trades'] = bt['n_trades']
    return m, sigs

def strat_lstm(trainer, dl, df, te_s, te_e, val_threshold=None, cost=10):
    """LSTM 策略 - 使用验证集阈值"""
    lb = (pd.to_datetime(te_s)-pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    probs, y, ret = trainer.predict(dl, lb, te_e)
    if probs is None: return None
    off = 60-30
    msk = (df.datetime>=pd.to_datetime(te_s))&(df.datetime<pd.to_datetime(te_e))
    n = min(len(probs)-off, max(0, msk.sum()-30))
    if n < 5: return None
    pt = probs[off:off+n]; rt = ret[off:off+n]
    
    # 使用验证集阈值（不在测试集上重新优化）
    if val_threshold is None:
        return None
    thr = val_threshold
    sigs = (pt >= thr).astype(int)
    bt = bt_long(np.array(sigs), rt, cost)
    m = calc_all_metrics(bt['net_returns']); m['trades']=bt['n_trades']; m['thr']=thr
    return m, sigs, pt

def strat_lstm_regime(trainer, dl, df, te_s, te_e, cost=10):
    """LSTM + 状态依赖阈值"""
    lb = (pd.to_datetime(te_s)-pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    probs, y, ret = trainer.predict(dl, lb, te_e)
    if probs is None: return None
    off = 60-30
    msk = (df.datetime>=pd.to_datetime(te_s))&(df.datetime<pd.to_datetime(te_e))
    n = min(len(probs)-off, max(0, msk.sum()-30))
    if n < 5: return None
    pt = probs[off:off+n]; rt = ret[off:off+n]
    test_idx = df[msk].index[30:30+n]; sigs = []
    for i, idx in enumerate(test_idx):
        r = regime(df.loc[idx])
        thr = {'trend_up':0.40, 'trend_down':0.50, 'chop':0.55, 'crisis':0.80}.get(r, 0.5)
        sigs.append(1 if pt[i] >= thr else 0)
    bt = bt_long(np.array(sigs), rt, cost)
    m = calc_all_metrics(bt['net_returns']); m['trades']=bt['n_trades']
    return m, sigs

FOLDS = [
    ("Fold 1","2018-07-01","2020-07-01","2020-07-01","2021-01-01","Recov"),
    ("Fold 2","2019-07-01","2021-07-01","2021-07-01","2022-01-01","Peak"),
    ("Fold 3","2020-07-01","2022-07-01","2022-07-01","2023-01-01","Crash"),
    ("Fold 4","2021-07-01","2023-07-01","2023-07-01","2024-01-01","Recov"),
    ("Fold 5","2022-01-01","2024-01-01","2024-01-01","2024-07-01","Bull"),
    ("Fold 6","2022-07-01","2024-07-01","2024-07-01","2025-01-01","Bull"),
    ("Fold 7","2023-01-01","2025-01-01","2025-01-01","2025-07-01","Bull"),
    ("Fold 8","2023-07-01","2025-07-01","2025-07-01","2026-01-01","Bear"),
]

def main():
    print("="*70); print("REGIME vs LSTM vs LSTM+REGIME (8-fold)"); print("="*70)
    df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
    df = prep_df(df)
    print(f"Data: {len(df)} rows\n"); results = []
    
    for fname, tr_s, tr_e, te_s, te_e, reg_label in FOLDS:
        df_fold = df.copy()
        
        msk = (df_fold.datetime>=pd.to_datetime(te_s))&(df_fold.datetime<pd.to_datetime(te_e))
        tr = df_fold.loc[msk,'return_1d'].values
        
        # Baselines
        close, s20, s50 = df_fold['price_close'], df_fold['sma20'], df_fold['sma50']
        sig_sma = (s20>s50).astype(int).loc[msk].values
        sma_m = calc_all_metrics(bt_long(sig_sma,tr)['net_returns'])
        bh_m = calc_all_metrics(tr)
        
        # Regime strategy
        rm, _ = strat_regime(df_fold, te_s, te_e)
        
        # LSTM
        bp = next((p for p in [
            f'btc_conditional/runs_v7/train_{tr_s[:10]}/best_model.pth',
            f'btc_conditional/runs_v7/train_{fname.replace(chr(32),chr(95))}/best_model.pth',
        ] if os.path.exists(p)), None)
        
        lm = lrm = None
        if bp:
            # 加载验证集阈值 + 归一化统计量
            val_thr, _, feat_mean, feat_std = load_val_threshold(bp)
            if val_thr is not None:
                t = BTCTrainer({'window':{'seq_len':30},'model':{'dropout':0.3},'trainer':{'lr':3e-4,'epochs':15}}, feats=ALL_FEATS)
                t.model.load_state_dict(torch.load(bp, map_location='cpu'))
                dl = BTCDataLoader(DATA_PATH, feats=ALL_FEATS)
                # 应用训练窗口归一化统计量
                if feat_mean is not None and feat_std is not None:
                    dl.apply_saved_norm(feat_mean, feat_std)
                else:
                    print(f"  WARNING: No normalization stats in {bp}!")
                lr = strat_lstm(t, dl, df_fold, te_s, te_e, val_threshold=val_thr)
                if lr: lm = lr[0]
                lr2 = strat_lstm_regime(t, dl, df_fold, te_s, te_e)
                if lr2: lrm = lr2[0]
            else:
                print(f"  No saved validation threshold, skipping LSTM")
        
        col = {'fold':fname,'regime':reg_label,'sma':sma_m['sharpe'],'bh':bh_m['sharpe'],
               'regime_sharpe':rm['sharpe'],'regime_trades':rm['trades'],
               'lstm_sharpe':lm['sharpe'] if lm else None,'lstm_trades':lm['trades'] if lm else None,
               'hybrid_sharpe':lrm['sharpe'] if lrm else None,'hybrid_trades':lrm['trades'] if lrm else None}
        
        lstm_s = f"{lm['sharpe']:+.3f}" if lm else "   N/A"
        hyb_s = f"{lrm['sharpe']:+.3f}" if lrm else "   N/A"
        print(f"{fname:<8} [{reg_label:>5}] Regime={rm['sharpe']:+.3f} LSTM={lstm_s:>8} Hybrid={hyb_s:>8} | SMA={sma_m['sharpe']:+.3f} BH={bh_m['sharpe']:+.3f}")
        results.append(col)
    
    # Summary
    print("\n"+"="*70); print("FINAL TABLE"); print("="*70)
    hdr = f"{'Fold':<8} {'Regime':>7} {'Regime_SR':>10} {'LSTM_SR':>8} {'Hybrid_SR':>10} {'SMA':>8} {'BH':>8}"
    print(hdr+"\n"+'-'*60)
    for r in results:
        print(f"{r['fold']:<8} {r['regime']:>7} {r['regime_sharpe']:>+10.3f} {r['lstm_sharpe'] or 0:>+8.3f} {r['hybrid_sharpe'] or 0:>+10.3f} {r['sma']:>+8.3f} {r['bh']:>+8.3f}")
    
    avg = lambda k: np.mean([r[k] for r in results if r[k] is not None])
    print(f"\n{'Average':<8} {'':>7} {avg('regime_sharpe'):>+10.3f} {avg('lstm_sharpe'):>+8.3f} {avg('hybrid_sharpe'):>+10.3f} {avg('sma'):>+8.3f} {avg('bh'):>+8.3f}")
    
    pd.DataFrame(results).to_csv(f'{RESULTS_DIR}/regime_final.csv', index=False)
    print(f"\nSaved to {RESULTS_DIR}/regime_final.csv")

if __name__ == "__main__":
    main()
