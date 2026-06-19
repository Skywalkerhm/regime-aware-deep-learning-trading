#!/usr/bin/env python3
"""综合分析：8折全部统计检验"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from scipy import stats
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_conditional.trainer import BTCTrainer, BTCDataLoader, ALL_FEATS
from btc_conditional.backtest import backtest
from btc_conditional.metrics import calc_all_metrics

DATA_PATH = "data/processed/btc_daily_full.csv"
RESULTS_DIR = "btc_conditional/results_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)

config = {"window": {"seq_len": 30}, "model": {"dropout": 0.3}, "trainer": {"lr": 3e-4, "epochs": 15}}

FOLDS = [
    ("Fold 1",  "2018-07-01","2020-07-01","2020-07-01","2021-01-01"),
    ("Fold 2",  "2019-07-01","2021-07-01","2021-07-01","2022-01-01"),
    ("Fold 3",  "2020-07-01","2022-07-01","2022-07-01","2023-01-01"),
    ("Fold 4",  "2021-07-01","2023-07-01","2023-07-01","2024-01-01"),
    ("Fold 5",  "2022-01-01","2024-01-01","2024-01-01","2024-07-01"),
    ("Fold 6",  "2022-07-01","2024-07-01","2024-07-01","2025-01-01"),
    ("Fold 7",  "2023-01-01","2025-01-01","2025-01-01","2025-07-01"),
    ("Fold 8",  "2023-07-01","2025-07-01","2025-07-01","2026-01-01"),
]
REGIMES = ["Recov", "Peak", "Crash", "Recov", "Bull", "Bull", "Bull", "Bear"]

def bt_long(sigs, ret, cost=10):
    return backtest(sigs, ret, cost_bps=cost)

def find_ckpt(fname, tr_s):
    for suffix in [f"train_{tr_s[:10]}", f"train_{fname.replace(' ','_')}"]:
        p = f"btc_conditional/runs_v7/{suffix}/best_model.pth"
        if os.path.exists(p):
            return p
    return None

def load_val_threshold(ckpt_path):
    """加载验证集上选定的阈值、温度和归一化统计量"""
    threshold_path = ckpt_path.replace("best_model.pth", "threshold.json")
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            data = json.load(f)
            return (data["threshold"], data.get("temperature", 1.0),
                    data.get("feat_mean"), data.get("feat_std"))
    return None, None, None, None

def main():
    print("=" * 80)
    print("8-FOLD COMPREHENSIVE EVALUATION")
    print("=" * 80)

    df_full = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    rng = np.random.RandomState(42)
    results = []

    for idx, (fname, tr_s, tr_e, te_s, te_e) in enumerate(FOLDS):
        regime = REGIMES[idx]
        print(f"\n{fname:<8} [{regime:>5}] {tr_s[:10]}~{tr_e[:10]} -> {te_s[:10]}~{te_e[:10]}")

        best_path = find_ckpt(fname, tr_s)
        if not best_path:
            print("  No checkpoint")
            continue

        trainer = BTCTrainer(config, feats=ALL_FEATS)
        trainer.model.load_state_dict(torch.load(best_path, map_location="cpu"))

        # 加载验证集阈值 + 归一化统计量
        val_thr, _, feat_mean, feat_std = load_val_threshold(best_path)
        if val_thr is None:
            print("  No saved validation threshold, skipping")
            continue
        print(f"  Val threshold: {val_thr:.3f}")

        # 每个 fold 创建新的 BTCDataLoader 并应用对应的归一化统计量
        dl = BTCDataLoader(DATA_PATH, feats=ALL_FEATS)
        if feat_mean is not None and feat_std is not None:
            dl.apply_saved_norm(feat_mean, feat_std)
        else:
            print(f"  WARNING: No normalization stats in {best_path}!")

        # Baselines
        mask = (df_full["datetime"] >= pd.to_datetime(te_s)) & \
               (df_full["datetime"] < pd.to_datetime(te_e))
        test_ret_raw = df_full.loc[mask, "return_1d"].values
        full_close = df_full["price_close"]
        sma20 = full_close.rolling(20).mean()
        sma50 = full_close.rolling(50).mean()
        bt = bt_long((sma20 > sma50).astype(int).loc[mask].values, test_ret_raw)
        m_sma = {**calc_all_metrics(bt["net_returns"]), "trades": bt["n_trades"]}
        m_bh = calc_all_metrics(test_ret_raw)
        bt = bt_long(rng.randint(0, 2, size=len(test_ret_raw)), test_ret_raw)
        m_rand = calc_all_metrics(bt["net_returns"])

        # LSTM test
        lb = (pd.to_datetime(te_s) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        probs, y_true, ret = trainer.predict(dl, lb, te_e)
        if probs is None:
            print("  No predictions")
            continue
        offset = 60 - 30
        test_n = mask.sum()
        n_test = min(len(probs) - offset, test_n - 30)
        if n_test < 5:
            print(f"  <5 test samples ({n_test})")
            continue
        probs_t = probs[offset:offset+n_test]
        ret_t = ret[offset:offset+n_test]

        # 使用验证集阈值
        thr = val_thr
        sigs = (probs_t >= thr).astype(int)
        lbt = bt_long(sigs, ret_t)
        m = calc_all_metrics(lbt["net_returns"])
        m["trades"] = lbt["n_trades"]
        nr = lbt["net_returns"]
        trade_mask = nr != 0
        m["win_rate"] = float((nr[trade_mask] > 0).mean()) if trade_mask.sum() > 0 else 0.0

        # Same-freq random
        k = lbt["n_trades"]
        n = len(ret_t)
        rand_p95 = None
        if 3 <= k < n - 3:
            sharpes = []
            for _ in range(2000):
                idxs = rng.choice(n, k, replace=False)
                rs = np.zeros(n, dtype=int); rs[idxs] = 1
                sharpes.append(calc_all_metrics(bt_long(rs, ret_t)["net_returns"])["sharpe"])
            rand_p95 = float(np.percentile(sharpes, 95))

        # Bootstrap CI
        ci_low, ci_high = None, None
        if len(lbt["net_returns"]) >= 10:
            boot = []
            for _ in range(2000):
                bs = np.array([])
                for _ in range(int(np.ceil(n/5))):
                    s = rng.randint(0, max(1, n-5))
                    bs = np.append(bs, lbt["net_returns"][s:s+5])
                bs = bs[:n]
                boot.append(calc_all_metrics(bs)["sharpe"])
            ci_low, ci_high = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

        # Deflated Sharpe Ratio
        ret_s = lbt["net_returns"]
        T = len(ret_s)
        sr = m["sharpe"]
        g3 = float(stats.skew(ret_s)) if T > 5 else 0
        g4 = float(stats.kurtosis(ret_s, fisher=True)) if T > 5 else 0
        sigma_sr = np.sqrt((1 + 0.5*sr**2 - g3*sr + (g4-1)/4*sr**2) / T) if T > 0 else 1
        N_trials = 20
        E_max = stats.norm.ppf(1 - 0.05/N_trials) * (1/np.sqrt(T)) if T > 0 else 0
        dsr_val = (sr - E_max) / sigma_sr if sigma_sr > 0 else 0
        dsr_sig = dsr_val > 1.645

        beats = m["sharpe"] > rand_p95 if rand_p95 is not None else None

        print(f"  LSTM: {m['sharpe']:+.3f} ({m['trades']}t) WinRate={m['win_rate']:.1%} | SMA: {m_sma['sharpe']:+.3f} BH: {m_bh['sharpe']:+.3f} Rand: {m_rand['sharpe']:+.3f}")
        rp95_s = f"{rand_p95:+.3f}" if rand_p95 is not None else "  N/A"
        ci_s = f"[{ci_low:+.1f},{ci_high:+.1f}]" if ci_low is not None else "N/A"
        print(f"  Rand-p95: {rp95_s} >p95? {beats} | CI95: {ci_s} | DSR={dsr_val:.2f} sig={dsr_sig}")
        lstm_sharpe_val = float(m["sharpe"])
        results.append({
            "fold": fname, "regime": regime,
            "lstm_sharpe": lstm_sharpe_val,
            "trades": m["trades"],
            "win_rate": float(m.get("win_rate", 0)),
            "sma_sharpe": float(m_sma["sharpe"]),
            "bh_sharpe": float(m_bh["sharpe"]),
            "rand_sharpe": float(m_rand["sharpe"]),
            "rand_p95": rand_p95,
            "beats_random": beats,
            "ci_low": ci_low, "ci_high": ci_high,
            "dsr": float(dsr_val), "dsr_sig": dsr_sig,
            "threshold": float(thr),
        })

    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    hdr = f"{'Fold':<8} {'Regime':>7} {'Sharpe':>8} {'T':>4} {'WinRate':>8} {'SMA':>8} {'BH':>8} {'Rnd95':>8} {'>p95':>5} {'CI(95)':>16} {'DSR':>6}"
    print(hdr)
    print("-" * 80)
    for r in results:
        ci = f"[{r['ci_low']:+.1f},{r['ci_high']:+.1f}]" if r['ci_low'] else "N/A"
        bp = "OK" if r['beats_random'] else "NO"
        wr = f"{r['win_rate']:.1%}" if r.get('win_rate') else "N/A"
        print(f"{r['fold']:<8} {r['regime']:>7} {r['lstm_sharpe']:>+8.3f} {r['trades']:>3d} {wr:>8} "
              f"{r['sma_sharpe']:>+8.3f} {r['bh_sharpe']:>+8.3f} "
              f"{r['rand_p95'] or 0:>+8.3f} {bp:>5} {ci:>16} {r['dsr']:+>6.2f}")

    s = [r["lstm_sharpe"] for r in results]
    bull_s = [r["lstm_sharpe"] for r in results if r["regime"] in ("Bull", "Peak", "Recov")]
    bear_s = [r["lstm_sharpe"] for r in results if r["regime"] in ("Bear", "Crash")]

    print(f"\nOverall:   {np.mean(s):+.3f} +/- {np.std(s, ddof=1):.3f}  (n={len(s)})")
    if bull_s: print(f"Bull:      {np.mean(bull_s):+.3f} +/- {np.std(bull_s, ddof=1):.3f}  (n={len(bull_s)})")
    if bear_s: print(f"Bear:      {np.mean(bear_s):+.3f} +/- {np.std(bear_s, ddof=1):.3f}  (n={len(bear_s)})")
    print(f"Beat Rnd:  {sum(1 for r in results if r['beats_random'])}/{len(results)}")
    print(f"DSR sig:   {sum(1 for r in results if r['dsr_sig'])}/{len(results)}")

    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "full_results.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump({"mean": float(np.mean(s)), "std": float(np.std(s, ddof=1))}, f)
    print(f"\nSaved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
