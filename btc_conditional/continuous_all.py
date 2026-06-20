"""连续仓位对比：Regime / LSTM / Hybrid — binary vs continuous"""
import os, sys, json, numpy as np, pandas as pd, torch
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btc_conditional.backtest import backtest
from btc_conditional.metrics import calc_all_metrics
from btc_conditional.trainer import BTCTrainer, BTCDataLoader, ALL_FEATS

DATA_PATH = "data/processed/btc_daily_full.csv"
RESULTS_DIR = "btc_conditional/results_v2"

def load_val_threshold(ckpt_path):
    """加载验证集上选定的阈值、温度和归一化统计量"""
    threshold_path = ckpt_path.replace("best_model.pth", "threshold.json")
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            data = json.load(f)
            return (data["threshold"], data.get("temperature", 1.0),
                    data.get("feat_mean"), data.get("feat_std"))
    return None, None, None, None

# ── Regime detection ──
def regime(row):
    v = row['volatility_20d']; sr = row['sma_ratio_20_50']; r20 = row['return_20d']
    if v > 0.186 or r20 < -0.10: return 'crisis'
    if abs(sr-1) > 0.008: return 'trend_up' if sr > 1 else 'trend_down'
    return 'chop'

# ── Continuous backtest ──
def bt_cont(positions, returns, cost_bps=10):
    """连续仓位回测 — 与 backtest() 保持一致的 1 天滞后：positions[t-1] → returns[t]"""
    cost = cost_bps/10000.0; prev=0.0; nr=np.zeros_like(returns); nt=0
    nr[0] = 0  # 第一天无前日信号
    for t in range(1, len(returns)):
        p=float(np.clip(positions[t-1],0,1)); ch=abs(p-prev); nt+=1 if ch>0.01 else 0
        nr[t]=p*returns[t]-ch*cost; prev=p
    m=calc_all_metrics(nr); m['trades']=nt; return m, nr

def bt_bin(sigs, ret, cost=10): bt=backtest(sigs,ret,cost_bps=cost); m=calc_all_metrics(bt['net_returns']); m['trades']=bt['n_trades']; return m

# ── Position functions ──
def pos_regime(df, idx):
    r = regime(df.loc[idx])
    if r == 'trend_up': return 1.0
    if r == 'trend_down': return 0.0
    if r == 'chop':
        rsi = df.loc[idx].get('rsi14', 50); rsi=50 if pd.isna(rsi) else rsi
        return float(np.clip((70-rsi)/35, 0, 1))
    return 0.0

def pos_lstm(prob): return float(np.clip(prob,0,1))

def pos_hybrid(prob, df, idx):
    r = regime(df.loc[idx])
    thr = {'trend_up': 0.40, 'trend_down': 0.50, 'chop': 0.55, 'crisis': 0.80}.get(r, 0.50)
    return float(prob >= thr)

# ── Preprocess ──
def prep(df):
    """
    预处理：计算技术指标
    
    SMA/RSI 是滚动指标，t 时刻只用过去值，不存在未来泄漏。
    直接对全量数据用 rolling 计算是安全的。
    """
    c = df['price_close']
    df['sma20'] = c.rolling(20).mean()
    df['sma50'] = c.rolling(50).mean()
    d = c.diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    df['rsi14'] = 100 - 100 / (1 + g / (l + 1e-10))
    return df

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
    print("="*70); print("CONTINUOUS POSITIONING: Binary vs Continuous (8-fold)"); print("="*70)
    df=pd.read_csv(DATA_PATH,parse_dates=['datetime'])
    df=prep(df)
    print(f"Data: {len(df)} rows\n")
    results=[]

    for fname,tr_s,tr_e,te_s,te_e,reg_label in FOLDS:
        df_fold = df.copy()
        
        msk=(df_fold.datetime>=pd.to_datetime(te_s))&(df_fold.datetime<pd.to_datetime(te_e))
        idx=df_fold[msk].index; tr=df_fold.loc[idx,'return_1d'].values
        close,s20,s50=df_fold['price_close'],df_fold['sma20'],df_fold['sma50']
        
        # ── Regime ──
        # Binary
        sigs=[];prev=0
        for i in idx:
            r=regime(df_fold.loc[i])
            if r=='trend_up': s=1 if df_fold.loc[i,'sma20']>df_fold.loc[i,'sma50'] else 0
            elif r=='trend_down': s=0
            elif r=='chop':
                rsi=df_fold.loc[i,'rsi14'];rsi=50 if pd.isna(rsi) else rsi
                s=1 if rsi<35 else(0 if rsi>65 else prev)
            else: s=0
            sigs.append(s);prev=s
        rm_b=bt_bin(np.array(sigs),tr)
        # Continuous
        poss=np.array([pos_regime(df_fold,i) for i in idx])
        rm_c,_=bt_cont(poss,tr)
        
        # ── LSTM ──
        bp=next((p for p in [f'btc_conditional/runs_v7/train_{tr_s[:10]}/best_model.pth',
                f'btc_conditional/runs_v7/train_{fname.replace(chr(32),chr(95))}/best_model.pth']
                if os.path.exists(p)),None)
        
        lm_b=lm_c=hm_b=hm_c=None
        if bp:
            # 加载验证集阈值 + 归一化统计量
            val_thr, _, feat_mean, feat_std = load_val_threshold(bp)
            if val_thr is not None:
                t=BTCTrainer({'window':{'seq_len':30},'model':{'dropout':0.3},'trainer':{'lr':3e-4,'epochs':15}},feats=ALL_FEATS)
                t.model.load_state_dict(torch.load(bp,map_location='cpu'))
                dl=BTCDataLoader(DATA_PATH,feats=ALL_FEATS)
                # 应用训练窗口归一化统计量
                if feat_mean is not None and feat_std is not None:
                    dl.apply_saved_norm(feat_mean, feat_std)
                else:
                    print(f"  WARNING: No normalization stats in {bp}!")
                lb=(pd.to_datetime(te_s)-pd.Timedelta(days=60)).strftime('%Y-%m-%d')
                probs,y,ret=t.predict(dl,lb,te_e)
                if probs is not None:
                    off=60-30;n=min(len(probs)-off,max(0,msk.sum()-30))
                    if n>=5:
                        pt=probs[off:off+n];rt=ret[off:off+n];ti=idx[30:30+n]
                        # LSTM binary - 使用验证集阈值
                        thr = val_thr
                        lm_b=bt_bin((pt>=thr).astype(int),rt)
                        # LSTM continuous
                        lpos=np.array([pos_lstm(p) for p in pt])
                        lm_c,_=bt_cont(lpos,rt)
                        # Hybrid binary
                        hs=np.array([1 if pt[j]>=(0.40 if regime(df_fold.loc[ti[j]])=='trend_up' else 0.5) else 0 for j in range(len(ti))])
                        hm_b=bt_bin(hs,rt)
                        # Hybrid continuous
                        hpos=np.array([pos_hybrid(pt[j],df_fold,ti[j]) for j in range(len(ti))])
                        hm_c,_=bt_cont(hpos,rt)
            else:
                print(f"  No saved validation threshold, skipping LSTM")
        
        r={'fold':fname,'regime':reg_label,
           'reg_bin_sharpe':rm_b['sharpe'],'reg_bin_trades':rm_b['trades'],
           'reg_cont_sharpe':rm_c['sharpe'],'reg_cont_trades':rm_c['trades'],
           'lstm_bin_sharpe':lm_b['sharpe'] if lm_b else None,'lstm_bin_trades':lm_b['trades'] if lm_b else None,
           'lstm_cont_sharpe':lm_c['sharpe'] if lm_c else None,'lstm_cont_trades':lm_c['trades'] if lm_c else None,
           'hyb_bin_sharpe':hm_b['sharpe'] if hm_b else None,'hyb_bin_trades':hm_b['trades'] if hm_b else None,
           'hyb_cont_sharpe':hm_c['sharpe'] if hm_c else None,'hyb_cont_trades':hm_c['trades'] if hm_c else None,
        }
        
        print(f"{fname:<8}[{reg_label:>5}] Reg: bin={rm_b['sharpe']:+.3f}({rm_b['trades']}t) cont={rm_c['sharpe']:+.3f}({rm_c['trades']}t)")
        print(f"         LSTM: bin={lm_b['sharpe'] if lm_b else 0:+.3f}({lm_b['trades'] if lm_b else 0}t) cont={lm_c['sharpe'] if lm_c else 0:+.3f}({lm_c['trades'] if lm_c else 0}t)")
        print(f"         Hyb:  bin={hm_b['sharpe'] if hm_b else 0:+.3f}({hm_b['trades'] if hm_b else 0}t) cont={hm_c['sharpe'] if hm_c else 0:+.3f}({hm_c['trades'] if hm_c else 0}t)")
        results.append(r)
    
    # Summary
    print("\n"+"="*70); print("FINAL: Binary vs Continuous Comparison"); print("="*70)
    hdr=f"{'Fold':<8} {'Reg-Bin':>14} {'Reg-Cont':>14} {'LSTM-Bin':>14} {'LSTM-Cont':>14} {'Hyb-Bin':>14} {'Hyb-Cont':>14}"
    print(hdr+"\n"+'-'*80)
    for r in results:
        rb=f"{r['reg_bin_sharpe']:+.2f}({r['reg_bin_trades']}t)"; rc=f"{r['reg_cont_sharpe']:+.2f}({r['reg_cont_trades']}t)"
        lb=f"{r['lstm_bin_sharpe'] or 0:+.2f}({r['lstm_bin_trades'] or 0}t)"; lc=f"{r['lstm_cont_sharpe'] or 0:+.2f}({r['lstm_cont_trades'] or 0}t)"
        hb=f"{r['hyb_bin_sharpe'] or 0:+.2f}({r['hyb_bin_trades'] or 0}t)"; hc=f"{r['hyb_cont_sharpe'] or 0:+.2f}({r['hyb_cont_trades'] or 0}t)"
        print(f"{r['fold']:<8} {rb:>14} {rc:>14} {lb:>14} {lc:>14} {hb:>14} {hc:>14}")
    
    avg=lambda k:np.mean([r[k] for r in results if r[k]is not None])
    print(f"\n{'Mean':<8} {avg('reg_bin_sharpe'):>+14.3f} {avg('reg_cont_sharpe'):>+14.3f} {avg('lstm_bin_sharpe'):>+14.3f} {avg('lstm_cont_sharpe'):>+14.3f} {avg('hyb_bin_sharpe'):>+14.3f} {avg('hyb_cont_sharpe'):>+14.3f}")
    print(f"{'Trades':<8} {avg('reg_bin_trades'):>14.0f} {avg('reg_cont_trades'):>14.0f} {avg('lstm_bin_trades'):>14.0f} {avg('lstm_cont_trades'):>14.0f} {avg('hyb_bin_trades'):>14.0f} {avg('hyb_cont_trades'):>14.0f}")
    
    pd.DataFrame(results).to_csv(f'{RESULTS_DIR}/continuous_all.csv',index=False)
    print(f"\nSaved to {RESULTS_DIR}/continuous_all.csv")

if __name__=="__main__":
    main()
