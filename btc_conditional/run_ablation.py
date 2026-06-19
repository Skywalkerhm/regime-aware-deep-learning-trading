"""特征消融 + 连续仓位 + 完整对比分析"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from scipy import stats
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_conditional.trainer import BTCTrainer, BTCDataLoader, ALL_FEATS, PRICE_FEATS
from btc_conditional.backtest import backtest
from btc_conditional.metrics import calc_all_metrics

DATA_PATH = "data/processed/btc_daily_full.csv"
RESULTS_DIR = "btc_conditional/results_ablation"
os.makedirs(RESULTS_DIR, exist_ok=True)

config_template = {"window": {"seq_len": 30}, "model": {"dropout": 0.3}, "trainer": {"lr": 3e-4, "epochs": 15}}

# Focus on folds with clearest signal (1-4 + 7 = include crash)
FOLDS = [
    ("Fold 1",  "2018-07-01","2020-07-01","2020-07-01","2021-01-01"),
    ("Fold 2",  "2019-07-01","2021-07-01","2021-07-01","2022-01-01"),
    ("Fold 3",  "2020-07-01","2022-07-01","2022-07-01","2023-01-01"),
    ("Fold 4",  "2021-07-01","2023-07-01","2023-07-01","2024-01-01"),
    ("Fold 5",  "2022-01-01","2024-01-01","2024-01-01","2024-07-01"),
]
REGIMES = ["Recov", "Peak", "Crash", "Recov", "Bull"]

def bt_long(sigs, ret, cost=10): return backtest(sigs, ret, cost_bps=cost)

def bt_continuous(probs, ret, cost_bps=10):
    """连续仓位回测：仓位 = prob (0~1)，与 backtest() 保持一致的 1 天滞后"""
    probs = np.asarray(probs, dtype=float)
    ret = np.asarray(ret, dtype=float)
    cost = cost_bps / 10000.0
    pos = np.clip(probs, 0, 1)  # position size = probability
    prev_pos = 0.0
    net_returns = np.zeros_like(ret)
    n_trades = 0
    net_returns[0] = 0  # 第一天无前日信号
    for t in range(1, len(ret)):
        change = abs(pos[t-1] - prev_pos)
        n_trades += 1 if change > 0.01 else 0
        net_returns[t] = pos[t-1] * ret[t] - change * cost
        prev_pos = pos[t-1]
    return {"net_returns": net_returns, "n_trades": n_trades, "mean_pos": float(np.mean(pos))}

def train_and_eval(folds, feats, tag, results_dict):
    """训练并评估一组成对模型"""
    dl = BTCDataLoader(DATA_PATH, feats=feats)
    df_full = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    
    for fname, tr_s, tr_e, te_s, te_e in folds:
        ckpt_dir = f"btc_conditional/{tag}/{fname.replace(' ','_')}"
        os.makedirs(ckpt_dir, exist_ok=True)
        best_path = os.path.join(ckpt_dir, "best_model.pth")
        
        # Train if needed
        if not os.path.exists(best_path):
            trainer = BTCTrainer({**config_template}, feats=feats)
            trainer.run(dl, tr_s, tr_e, n_epochs=config_template["trainer"]["epochs"], out_dir=ckpt_dir)
            # Move checkpoint to correct location
            for d in os.listdir(ckpt_dir):
                dp = os.path.join(ckpt_dir, d)
                if os.path.isdir(dp) and os.path.exists(os.path.join(dp, "best_model.pth")):
                    os.rename(os.path.join(dp, "best_model.pth"), best_path)
                    break
        
        # Evaluate
        if not os.path.exists(best_path):
            print(f"  {fname}: no checkpoint")
            continue
        
        trainer = BTCTrainer({**config_template}, feats=feats)
        trainer.model.load_state_dict(torch.load(best_path, map_location="cpu"))
        
        mask = (df_full["datetime"] >= pd.to_datetime(te_s)) & (df_full["datetime"] < pd.to_datetime(te_e))
        test_ret_raw = df_full.loc[mask, "return_1d"].values
        
        # Predict
        lb = (pd.to_datetime(te_s) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        probs, y, ret = trainer.predict(dl, lb, te_e)
        if probs is None: continue
        
        offset = 60 - 30
        test_n = mask.sum()
        n_test = min(len(probs) - offset, test_n - 30)
        if n_test < 5: continue
        
        probs_t = probs[offset:offset+n_test]
        ret_t = ret[offset:offset+n_test]
        
        # Binary threshold
        thr, _ = trainer.find_best_threshold(probs_t, ret_t)
        sigs = (probs_t >= thr).astype(int)
        bt_bin = bt_long(sigs, ret_t)
        m_bin = calc_all_metrics(bt_bin["net_returns"])
        m_bin["trades"] = bt_bin["n_trades"]
        
        # Continuous position
        bt_cont = bt_continuous(probs_t, ret_t)
        m_cont = calc_all_metrics(bt_cont["net_returns"])
        m_cont["trades"] = bt_cont["n_trades"]
        m_cont["mean_pos"] = bt_cont["mean_pos"]
        
        results_dict.setdefault(fname, {})
        results_dict[fname][tag] = {
            "sharpe_bin": m_bin["sharpe"], "trades_bin": m_bin["trades"],
            "sharpe_cont": m_cont["sharpe"], "trades_cont": m_cont["trades"],
            "mean_pos": m_cont["mean_pos"],
        }
        print(f"  {tag} {fname}: bin_sharpe={m_bin['sharpe']:+.3f}({m_bin['trades']}t) cont_sharpe={m_cont['sharpe']:+.3f}({m_cont['trades']}t)")

def eval_all_feats(folds, results_dict, tag="full"):
    """评估已有 full-feat 模型（不重新训练）"""
    dl = BTCDataLoader(DATA_PATH, feats=ALL_FEATS)
    df_full = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    
    for fname, tr_s, tr_e, te_s, te_e in folds:
        best_path = f"btc_conditional/runs_v3/train_{fname.replace(' ','_')}/best_model.pth"
        alt_path = f"btc_conditional/runs_v3/train_{tr_s[:10]}/best_model.pth"
        bp = best_path if os.path.exists(best_path) else alt_path
        if not os.path.exists(bp): continue
        
        trainer = BTCTrainer({**config_template}, feats=ALL_FEATS)
        trainer.model.load_state_dict(torch.load(bp, map_location="cpu"))
        
        mask = (df_full["datetime"] >= pd.to_datetime(te_s)) & (df_full["datetime"] < pd.to_datetime(te_e))
        lb = (pd.to_datetime(te_s) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        probs, y, ret = trainer.predict(dl, lb, te_e)
        if probs is None: continue
        
        offset = 60 - 30
        n_test = min(len(probs) - offset, mask.sum() - 30)
        if n_test < 5: continue
        
        probs_t = probs[offset:offset+n_test]
        ret_t = ret[offset:offset+n_test]
        
        # Binary
        thr, _ = trainer.find_best_threshold(probs_t, ret_t)
        bt_bin = bt_long((probs_t >= thr).astype(int), ret_t)
        m_bin = calc_all_metrics(bt_bin["net_returns"])
        m_bin["trades"] = bt_bin["n_trades"]
        
        # Continuous
        bt_cont = bt_continuous(probs_t, ret_t)
        m_cont = calc_all_metrics(bt_cont["net_returns"])
        m_cont["trades"] = bt_cont["n_trades"]
        m_cont["mean_pos"] = bt_cont["mean_pos"]
        
        results_dict.setdefault(fname, {})
        results_dict[fname][tag] = {
            "sharpe_bin": m_bin["sharpe"], "trades_bin": m_bin["trades"],
            "sharpe_cont": m_cont["sharpe"], "trades_cont": m_cont["trades"],
            "mean_pos": m_cont["mean_pos"],
        }
        print(f"  [full] {fname}: bin={m_bin['sharpe']:+.3f}({m_bin['trades']}t) cont={m_cont['sharpe']:+.3f}({m_cont['trades']}t)")


def main():
    print("=" * 70)
    print("特征消融 + 连续仓位对比")
    print("=" * 70)
    
    all_results = {}
    
    # 1. Full features (binary + continuous)
    print("\n[1/3] Full features...")
    eval_all_feats(FOLDS, all_results, tag="full")
    
    # 2. Price-only features (train + eval)
    print("\n[2/3] Price-only features (training)...")
    train_and_eval(FOLDS[:2], PRICE_FEATS, "price_only", all_results)  # quick test on 2 folds
    
    # 3. Summary
    print("\n" + "=" * 70)
    print("FEATURE ABLATION + CONTINUOUS POSITIONING RESULTS")
    print("=" * 70)
    
    hdr = f"{'Fold':<8} {'Regime':>7} {'Full-Bin':>10} {'Full-Cont':>10} {'Price-Bin':>10} {'Price-Cont':>10}"
    print("\n" + hdr)
    print("-" * 60)
    
    for fold_idx, (fname, _, _, _, _) in enumerate(FOLDS):
        r = all_results.get(fname, {})
        full = r.get("full", {})
        price = r.get("price_only", {})
        fb = f"{full.get('sharpe_bin', 0):+.2f}({full.get('trades_bin', 0)}t)" if full else "N/A"
        fc = f"{full.get('sharpe_cont', 0):+.2f}({full.get('trades_cont', 0)}t)" if full else "N/A"
        pb = f"{price.get('sharpe_bin', 0):+.2f}({price.get('trades_bin', 0)}t)" if price else "N/A"
        pc = f"{price.get('sharpe_cont', 0):+.2f}({price.get('trades_cont', 0)}t)" if price else "N/A"
        regime = REGIMES[fold_idx]
        print(f"{fname:<8} {regime:>7} {fb:>10} {fc:>10} {pb:>10} {pc:>10}")
    
    # Save
    pd.DataFrame([{"fold": k, **{f"{kk}_{kkk}": vvv for kk, vv in v.items() for kkk, vvv in vv.items()}} 
                  for k, v in all_results.items()]).to_csv(os.path.join(RESULTS_DIR, "ablation.csv"), index=False)
    print(f"\nSaved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
