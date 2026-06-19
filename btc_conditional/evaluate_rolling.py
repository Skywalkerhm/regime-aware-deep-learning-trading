"""滚动窗口评估 — 每个 6 个月测试期使用对应的滚动窗口模型"""
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
RUNS_DIR = "btc_conditional/runs_v7"

# 滚动窗口配置（与训练一致）
ROLLING_WINDOWS = [
    ("2018-07-01", "2020-07-01", "2020-07-01", "2021-01-01", "roll_2020-07", "Recov"),
    ("2019-01-01", "2021-01-01", "2021-01-01", "2021-07-01", "roll_2021-01", "Peak"),
    ("2019-07-01", "2021-07-01", "2021-07-01", "2022-01-01", "roll_2021-07", "Peak"),
    ("2020-01-01", "2022-01-01", "2022-01-01", "2022-07-01", "roll_2022-01", "Crash"),
    ("2020-07-01", "2022-07-01", "2022-07-01", "2023-01-01", "roll_2022-07", "Bear"),
    ("2021-01-01", "2023-01-01", "2023-01-01", "2023-07-01", "roll_2023-01", "Recov"),
    ("2021-07-01", "2023-07-01", "2023-07-01", "2024-01-01", "roll_2023-07", "Recov"),
    ("2022-01-01", "2024-01-01", "2024-01-01", "2024-07-01", "roll_2024-01", "Bull"),
    ("2022-07-01", "2024-07-01", "2024-07-01", "2025-01-01", "roll_2024-07", "Bull"),
    ("2023-01-01", "2025-01-01", "2025-01-01", "2025-07-01", "roll_2025-01", "Bull"),
    ("2023-07-01", "2025-07-01", "2025-07-01", "2026-01-01", "roll_2025-07", "Bear"),
]

config = {"window": {"seq_len": 30}, "model": {"dropout": 0.3}, "trainer": {"lr": 3e-4, "epochs": 15}}

def load_val_threshold(ckpt_path):
    """加载验证集上选定的阈值、温度和归一化统计量"""
    threshold_path = ckpt_path.replace("best_model.pth", "threshold.json")
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            data = json.load(f)
            return (data["threshold"], data.get("temperature", 1.0),
                    data.get("feat_mean"), data.get("feat_std"))
    return None, None, None, None

def bt_long(sigs, ret, cost=10):
    return backtest(sigs, ret, cost_bps=cost)

def main():
    print("=" * 80)
    print("滚动窗口评估 (2 年历史, 每 6 个月重新训练)")
    print("=" * 80)

    df_full = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    results = []

    for tr_s, tr_e, te_s, te_e, roll_name, regime_label in ROLLING_WINDOWS:
        print(f"\n{roll_name:<15} [{regime_label:>5}] {tr_s[:10]}~{tr_e[:10]} -> {te_s[:10]}~{te_e[:10]}")

        # 加载滚动窗口模型
        ckpt_path = os.path.join(RUNS_DIR, roll_name, "best_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"  No checkpoint: {ckpt_path}")
            continue

        trainer = BTCTrainer(config, feats=ALL_FEATS)
        trainer.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        # 加载验证集阈值 + 归一化统计量
        val_thr, _, feat_mean, feat_std = load_val_threshold(ckpt_path)
        if val_thr is None:
            print("  No saved validation threshold")
            continue
        print(f"  Val threshold: {val_thr:.3f}")

        # 每个 fold 创建新的 BTCDataLoader 并应用对应的归一化统计量
        dl = BTCDataLoader(DATA_PATH, feats=ALL_FEATS)
        if feat_mean is not None and feat_std is not None:
            dl.apply_saved_norm(feat_mean, feat_std)
        else:
            print(f"  WARNING: No normalization stats in {ckpt_path}!")

        # 预测
        lb = (pd.to_datetime(te_s) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        probs, y_true, ret = trainer.predict(dl, lb, te_e)
        if probs is None:
            print("  No predictions")
            continue

        # 提取测试期数据
        offset = 60 - 30
        mask = (df_full["datetime"] >= pd.to_datetime(te_s)) & \
               (df_full["datetime"] < pd.to_datetime(te_e))
        test_n = mask.sum()
        n_test = min(len(probs) - offset, test_n - 30)
        if n_test < 5:
            print(f"  <5 test samples ({n_test})")
            continue

        probs_t = probs[offset:offset+n_test]
        ret_t = ret[offset:offset+n_test]

        # 使用验证集阈值（不在测试集上优化）
        sigs = (probs_t >= val_thr).astype(int)
        lbt = bt_long(sigs, ret_t)
        m = calc_all_metrics(lbt["net_returns"])
        m["trades"] = lbt["n_trades"]
        
        # 计算测试集胜率
        nr = lbt["net_returns"]
        trade_mask = nr != 0
        m["win_rate"] = float((nr[trade_mask] > 0).mean()) if trade_mask.sum() > 0 else 0.0

        print(f"  LSTM: Sharpe={m['sharpe']:+.3f} ({m['trades']}t) WinRate={m['win_rate']:.1%}")

        results.append({
            "window": roll_name,
            "regime": regime_label,
            "train_period": f"{tr_s[:10]}~{tr_e[:10]}",
            "test_period": f"{te_s[:10]}~{te_e[:10]}",
            "val_threshold": val_thr,
            "sharpe": m["sharpe"],
            "trades": m["trades"],
            "win_rate": m["win_rate"],
        })

    # 汇总
    print("\n" + "=" * 80)
    print("滚动窗口汇总")
    print("=" * 80)
    hdr = f"{'Window':<15} {'Regime':>6} {'Val_Thr':>8} {'Sharpe':>8} {'Trades':>7} {'WinRate':>8}"
    print(hdr)
    print("-" * 60)
    for r in results:
        print(f"{r['window']:<15} {r['regime']:>6} {r['val_threshold']:>8.3f} "
              f"{r['sharpe']:>+8.3f} {r['trades']:>7d} {r['win_rate']:>7.1%}")

    # 统计
    sharpes = [r["sharpe"] for r in results]
    trades = [r["trades"] for r in results]
    win_rates = [r["win_rate"] for r in results if r["trades"] > 0]
    
    print(f"\n平均 Sharpe: {np.mean(sharpes):+.3f} +/- {np.std(sharpes):.3f}")
    print(f"平均交易数: {np.mean(trades):.1f}")
    if win_rates:
        print(f"平均胜率: {np.mean(win_rates):.1%}")

    # 保存
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/rolling_window.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/rolling_window.csv")

if __name__ == "__main__":
    main()
