"""Hybrid 策略有效性全面验证 — 论文级别分析"""
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

DATA_PATH = "data/processed/btc_daily_full.csv"
RESULTS_DIR = "btc_conditional/results_v2"

# ── 状态检测 (与 regime_final.py 一致) ──
THRESHOLDS = {'vol_p75': 0.093, 'crisis_ret': -0.10, 'sma_tight': 0.008, 'sma_wide': 0.025}

def regime(df_row):
    """Regime detection - 与 regime_final.py 一致"""
    v = df_row['volatility_20d']
    sr = df_row['sma_ratio_20_50']
    r20 = df_row['return_20d']
    
    if v > THRESHOLDS['vol_p75']*2 or r20 < THRESHOLDS['crisis_ret']:
        return 'crisis'
    if abs(sr-1) > THRESHOLDS['sma_wide']:
        return 'trend_up' if sr > 1 else 'trend_down'
    if abs(sr-1) > THRESHOLDS['sma_tight']:
        return 'trend_up' if sr > 1 else 'trend_down'
    return 'chop'

def bt_long(sigs, ret, cost=10):
    return backtest(sigs, ret, cost_bps=cost)

def prep_df(df):
    """预处理：计算技术指标 - 与 regime_final.py 一致"""
    close = df['price_close']
    df['sma20'] = close.rolling(20).mean()
    df['sma50'] = close.rolling(50).mean()
    delta = close.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi14'] = 100 - 100 / (1 + g / (l + 1e-10))
    return df

def load_norm_stats(ckpt_path):
    """从 threshold.json 加载训练窗口归一化统计量"""
    thr_path = ckpt_path.replace("best_model.pth", "threshold.json")
    if os.path.exists(thr_path):
        with open(thr_path, "r") as f:
            data = json.load(f)
            if "feat_mean" in data and "feat_std" in data:
                return data["feat_mean"], data["feat_std"]
    return None, None

def main():
    print("=" * 100)
    print("HYBRID STRATEGY VALIDATION — 论文级别分析")
    print("=" * 100)
    
    # 加载数据
    df_full = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df_full = prep_df(df_full)
    
    # FOLDS 定义
    FOLDS = [
        ("Fold 1", "2018-07-01", "2020-07-01", "2020-07-01", "2021-01-01"),
        ("Fold 2", "2019-07-01", "2021-07-01", "2021-07-01", "2022-01-01"),
        ("Fold 3", "2020-07-01", "2022-07-01", "2022-07-01", "2023-01-01"),
        ("Fold 4", "2021-07-01", "2023-07-01", "2023-07-01", "2024-01-01"),
        ("Fold 5", "2022-01-01", "2024-01-01", "2024-01-01", "2024-07-01"),
        ("Fold 6", "2022-07-01", "2024-07-01", "2024-07-01", "2025-01-01"),
        ("Fold 7", "2023-01-01", "2025-01-01", "2025-01-01", "2025-07-01"),
        ("Fold 8", "2023-07-01", "2025-07-01", "2025-07-01", "2026-01-01"),
    ]
    REGIMES = ["Recov", "Peak", "Crash", "Recov", "Bull", "Bull", "Bull", "Bear"]
    
    config = {"window": {"seq_len": 30}, "model": {"dropout": 0.3}, "trainer": {"lr": 3e-4, "epochs": 15}}
    
    results = []
    
    for idx, (fname, tr_s, tr_e, te_s, te_e) in enumerate(FOLDS):
        reg_label = REGIMES[idx]
        df_fold = df_full[(df_full["datetime"] >= pd.to_datetime(tr_s)) & 
                          (df_full["datetime"] < pd.to_datetime(te_e))].copy()
        
        print(f"\n{fname} [{reg_label}] {te_s[:10]}~{te_e[:10]}")
        print("-" * 80)
        
        # ── Buy & Hold ──
        mask_bh = (df_fold.datetime >= pd.to_datetime(te_s)) & (df_fold.datetime < pd.to_datetime(te_e))
        ret_bh = df_fold[mask_bh]['return_1d'].values
        cum_bh = np.cumprod(1 + ret_bh) - 1
        total_ret_bh = cum_bh[-1] if len(cum_bh) > 0 else 0
        sharpe_bh = np.mean(ret_bh) / np.std(ret_bh) * np.sqrt(252) if np.std(ret_bh) > 0 else 0
        
        # ── SMA (SMA20/SMA50 crossover — 与 regime_final.py 一致) ──
        closes = df_fold['price_close'].values
        sma20 = pd.Series(closes).rolling(20).mean().values
        sma50 = pd.Series(closes).rolling(50).mean().values
        sigs_sma = (sma20 > sma50).astype(int)
        ret_sma = df_fold['return_1d'].values

        # Fix: evaluate SMA only on test period (consistent with other baselines)
        test_mask_sma = df_fold.iloc[30:]['datetime'] >= pd.to_datetime(te_s)
        sigs_sma_te = sigs_sma[30:][test_mask_sma.values]
        ret_sma_te  = ret_sma[30:][test_mask_sma.values]
        bt_sma = bt_long(sigs_sma_te, ret_sma_te)
        sma_m = calc_all_metrics(bt_sma['net_returns'])
        sma_m['trades'] = bt_sma['n_trades']
        
        # ── Regime ──
        # 简单规则：趋势向上时买入
        regime_signals = []
        for i in range(30, len(df_fold)):
            row = df_fold.iloc[i]
            r = regime(row)
            regime_signals.append(1 if r == 'trend_up' else 0)
        regime_signals = np.array(regime_signals)
        bt_regime = bt_long(regime_signals, ret_sma[30:])
        rm = calc_all_metrics(bt_regime['net_returns'])
        rm['trades'] = bt_regime['n_trades']
        
        # ── LSTM ──
        bp = next((p for p in [
            f'btc_conditional/runs_v7/train_{tr_s[:10]}/best_model.pth',
            f'btc_conditional/runs_v7/train_{fname.replace(" ","_")}/best_model.pth',
        ] if os.path.exists(p)), None)
        
        lstm_sharpe = 0.0
        lstm_trades = 0
        lstm_cumret = 0.0
        lstm_probs = None
        lstm_ret = None
        test_idx = None
        
        if bp:
            trainer = BTCTrainer(config, feats=ALL_FEATS)
            trainer.model.load_state_dict(torch.load(bp, map_location='cpu'))
            dl = BTCDataLoader(DATA_PATH, feats=ALL_FEATS)
            
            # 加载训练窗口归一化统计量并应用
            feat_mean, feat_std = load_norm_stats(bp)
            if feat_mean is not None:
                dl.apply_saved_norm(feat_mean, feat_std)
            else:
                print(f"  WARNING: No normalization stats found in {bp}, results may be invalid!")
            
            lb = (pd.to_datetime(te_s) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
            probs, y, ret = trainer.predict(dl, lb, te_e)
            
            if probs is not None:
                off = 60 - 30
                msk = (df_fold.datetime >= pd.to_datetime(te_s)) & (df_fold.datetime < pd.to_datetime(te_e))
                n = min(len(probs) - off, max(0, msk.sum() - 30))
                if n >= 5:
                    pt = probs[off:off+n]
                    rt = ret[off:off+n]
                    # 使用测试期的 index - 与 regime_final.py 一致
                    ti = df_fold[msk].index[30:30+n]
                    
                    lstm_probs = pt
                    lstm_ret = rt
                    test_idx = ti
                    
                    # LSTM binary
                    thr_path = bp.replace("best_model.pth", "threshold.json")
                    val_thr = 0.5
                    if os.path.exists(thr_path):
                        with open(thr_path, "r") as f:
                            val_thr = json.load(f)["threshold"]
                    
                    sigs_lstm = (pt >= val_thr).astype(int)
                    bt_lstm = bt_long(sigs_lstm, rt)
                    lm = calc_all_metrics(bt_lstm['net_returns'])
                    lstm_sharpe = lm['sharpe']
                    lstm_trades = bt_lstm['n_trades']
                    lstm_cumret = float(np.prod(1 + bt_lstm['net_returns']) - 1)
        
        # ── Hybrid (LSTM + Regime) ──
        hyb_sharpe = 0.0
        hyb_trades = 0
        hyb_cumret = 0.0
        
        if lstm_probs is not None and len(lstm_probs) > 0:
            hyb_signals = []
            for j in range(len(test_idx)):
                prob = lstm_probs[j]
                # 使用 df_fold 做 regime 检测
                row = df_fold.loc[test_idx[j]]
                r = regime(row)
                # 状态依赖阈值 - 与 regime_final.py 一致
                thr = {'trend_up':0.40, 'trend_down':0.50, 'chop':0.55, 'crisis':0.80}.get(r, 0.5)
                hyb_signals.append(1 if prob >= thr else 0)
            
            hyb_signals = np.array(hyb_signals)
            bt_hyb = bt_long(hyb_signals, lstm_ret)
            hm = calc_all_metrics(bt_hyb['net_returns'])
            hyb_sharpe = hm['sharpe']
            hyb_trades = bt_hyb['n_trades']
            hyb_cumret = float(np.prod(1 + bt_hyb['net_returns']) - 1)
        
        # ── 打印结果 ──
        print(f"  Buy&Hold:   Sharpe={sharpe_bh:+.3f}  Ret={total_ret_bh:+.1%}")
        print(f"  SMA:        Sharpe={sma_m['sharpe']:+.3f}  Trades={sma_m['trades']}")
        print(f"  Regime:     Sharpe={rm['sharpe']:+.3f}  Trades={rm['trades']}")
        print(f"  LSTM:       Sharpe={lstm_sharpe:+.3f}  Trades={lstm_trades}  Ret={lstm_cumret:+.1%}")
        print(f"  Hybrid:     Sharpe={hyb_sharpe:+.3f}  Trades={hyb_trades}  Ret={hyb_cumret:+.1%}")
        
        improvement = hyb_sharpe - lstm_sharpe
        print(f"  Improvement (Hybrid-LSTM): {improvement:+.3f}")
        
        results.append({
            'fold': fname,
            'regime': reg_label,
            'bh_sharpe': sharpe_bh,
            'bh_return': total_ret_bh,
            'sma_sharpe': sma_m['sharpe'],
            'sma_trades': sma_m['trades'],
            'regime_sharpe': rm['sharpe'],
            'regime_trades': rm['trades'],
            'lstm_sharpe': lstm_sharpe,
            'lstm_trades': lstm_trades,
            'lstm_return': lstm_cumret,
            'hybrid_sharpe': hyb_sharpe,
            'hybrid_trades': hyb_trades,
            'hybrid_return': hyb_cumret,
            'improvement': improvement,
        })
    
    # ════════════════════════════════════════════════════════════════════
    # 汇总分析
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Fold':<8} {'Regime':>6} | {'BH_SR':>7} {'SMA_SR':>7} {'Reg_SR':>7} {'LSTM_SR':>8} {'Hyb_SR':>8} | {'Improve':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r['fold']:<8} {r['regime']:>6} | {r['bh_sharpe']:>+7.3f} {r['sma_sharpe']:>+7.3f} {r['regime_sharpe']:>+7.3f} {r['lstm_sharpe']:>+8.3f} {r['hybrid_sharpe']:>+8.3f} | {r['improvement']:>+8.3f}")
    
    print("-" * 90)
    
    # 计算平均值
    avg_bh = np.mean([r['bh_sharpe'] for r in results])
    avg_sma = np.mean([r['sma_sharpe'] for r in results])
    avg_reg = np.mean([r['regime_sharpe'] for r in results])
    avg_lstm = np.mean([r['lstm_sharpe'] for r in results])
    avg_hyb = np.mean([r['hybrid_sharpe'] for r in results])
    avg_imp = np.mean([r['improvement'] for r in results])
    
    print(f"{'Average':<8} {'':>6} | {avg_bh:>+7.3f} {avg_sma:>+7.3f} {avg_reg:>+7.3f} {avg_lstm:>+8.3f} {avg_hyb:>+8.3f} | {avg_imp:>+8.3f}")
    
    # ════════════════════════════════════════════════════════════════════
    # Hybrid vs LSTM 配对检验
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("STATISTICAL TESTS: Hybrid vs LSTM")
    print("=" * 100)
    
    lstm_sharpes = np.array([r['lstm_sharpe'] for r in results])
    hyb_sharpes = np.array([r['hybrid_sharpe'] for r in results])
    improvements = hyb_sharpes - lstm_sharpes
    
        # Bootstrap test (non-parametric, n=10000 resamples)
    n_boot = 10000
    rng = np.random.RandomState(42)
    n_folds = len(improvements)
    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n_folds, size=n_folds)
        boot_means[b] = np.mean(improvements[idx])
    
    boot_ci_low = float(np.percentile(boot_means, 2.5))
    boot_ci_high = float(np.percentile(boot_means, 97.5))
    boot_pval = (np.sum(boot_means <= 0) + 1) / (n_boot + 1)
    
    print(f"\\nBootstrap test (n={n_boot} resamples):")
    print(f"  Mean Improvement:    {np.mean(improvements):+.3f}")
    print(f"  95% Bootstrap CI:    [{boot_ci_low:+.3f}, {boot_ci_high:+.3f}]")
    print(f"  Bootstrap p-value:   {boot_pval:.4f}")
    print(f"  Significant (p<0.05)? {'YES' if boot_pval < 0.05 else 'NO'}")
    
    # Paired t-test (parametric, for comparison)
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(hyb_sharpes, lstm_sharpes)
    print(f"\\nPaired t-test (Hybrid vs LSTM):")
    print(f"  Mean Hybrid Sharpe:  {np.mean(hyb_sharpes):+.3f} +/- {np.std(hyb_sharpes, ddof=1):.3f}")
    print(f"  Mean LSTM Sharpe:    {np.mean(lstm_sharpes):+.3f} +/- {np.std(lstm_sharpes, ddof=1):.3f}")
    print(f"  Mean Improvement:    {np.mean(improvements):+.3f}")
    print(f"  t-statistic:         {t_stat:.3f}")
    print(f"  p-value (two-tail):  {p_val:.4f}")
    print(f"  Significant (p<0.05)? {'YES' if p_val < 0.05 else 'NO'}")
    
    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, w_pval = stats.wilcoxon(hyb_sharpes, lstm_sharpes)
        print(f"\\nWilcoxon signed-rank test:")
        print(f"  W-statistic:         {w_stat:.3f}")
        print(f"  p-value:             {w_pval:.4f}")
        print(f"  Significant (p<0.05)? {'YES' if w_pval < 0.05 else 'NO'}")
    except:
        pass
    
    # Summary
    boot_sig = boot_pval < 0.05
    t_sig = p_val < 0.05
    print(f"\\n  Bootstrap & t-test agree? {'YES' if boot_sig == t_sig else 'Check individual results above'}")
    
# ════════════════════════════════════════════════════════════════════
    # Hybrid vs 所有策略对比
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("STRATEGY RANKING (by mean Sharpe)")
    print("=" * 100)
    
    strategies = {
        'Hybrid (LSTM+Regime)': avg_hyb,
        'LSTM (pure)': avg_lstm,
        'Buy & Hold': avg_bh,
        'SMA': avg_sma,
        'Regime (pure rule)': avg_reg,
    }
    
    ranked = sorted(strategies.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, sharpe) in enumerate(ranked, 1):
        marker = " <-- TARGET" if name == 'Hybrid (LSTM+Regime)' else ""
        print(f"  #{rank}: {name:<25} Sharpe = {sharpe:+.3f}{marker}")
    
    # ════════════════════════════════════════════════════════════════════
    # Hybrid 在各市场状态的表现
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("HYBRID PERFORMANCE BY MARKET REGIME")
    print("=" * 100)
    
    regime_groups = {}
    for r in results:
        reg = r['regime']
        if reg not in regime_groups:
            regime_groups[reg] = []
        regime_groups[reg].append(r)
    
    for reg in ['Recov', 'Peak', 'Crash', 'Bull', 'Bear']:
        if reg in regime_groups:
            group = regime_groups[reg]
            avg_hyb_sr = np.mean([g['hybrid_sharpe'] for g in group])
            avg_lstm_sr = np.mean([g['lstm_sharpe'] for g in group])
            avg_bh_sr = np.mean([g['bh_sharpe'] for g in group])
            print(f"  {reg:>6}: Hybrid={avg_hyb_sr:+.3f}  LSTM={avg_lstm_sr:+.3f}  BH={avg_bh_sr:+.3f}  (n={len(group)})")
    
    # 保存结果
    df_results = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_results.to_csv(f"{RESULTS_DIR}/hybrid_validation.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/hybrid_validation.csv")

if __name__ == "__main__":
    main()

