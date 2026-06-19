#!/usr/bin/env python3
"""BTC LSTM — 滚动窗口验证 + 基线对比（标准训练）"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import warnings; warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from btc_conditional.trainer import BTCTrainer, BTCDataLoader, ALL_FEATS, PRICE_FEATS
from btc_conditional.backtest import backtest
from btc_conditional.metrics import calc_all_metrics

DATA_PATH = "data/processed/btc_daily_full.csv"
RESULTS_DIR = "btc_conditional/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

config = {
    "window": {"seq_len": 30},
    "model": {"dropout": 0.3},
    "trainer": {"lr": 3e-4, "epochs": 15},
}

FOLDS = [
    ("Fold 1", "2022-01-01", "2024-01-01", "2024-01-01", "2024-07-01"),
    ("Fold 2", "2022-07-01", "2024-07-01", "2024-07-01", "2025-01-01"),
    ("Fold 3", "2023-01-01", "2025-01-01", "2025-01-01", "2025-07-01"),
    ("Fold 4", "2023-07-01", "2025-07-01", "2025-07-01", "2026-01-01"),
]


def run_baselines(df):
    """
    计算4折的基线策略Sharpe
    
    SMA 是滚动指标，t 时刻只用过去值，不存在未来泄漏。
    统一使用全量数据 rolling 计算（与 analyze_all.py 口径一致）。
    """
    results = []
    for fname, tr_s, tr_e, te_s, te_e in FOLDS:
        test_mask = (df["datetime"] >= pd.to_datetime(te_s)) & \
                    (df["datetime"] < pd.to_datetime(te_e))
        test_ret = df.loc[test_mask, "return_1d"].values
        if len(test_ret) < 10:
            continue

        # 统一口径：全量 rolling（SMA 只看过去，无未来泄漏）
        full_close = df["price_close"]
        full_sma20 = full_close.rolling(20).mean()
        full_sma50 = full_close.rolling(50).mean()
        sma_sig = (full_sma20 > full_sma50).astype(int).loc[test_mask].values
        
        bt_sma = backtest(sma_sig, test_ret, cost_bps=10)
        m_sma = calc_all_metrics(bt_sma["net_returns"])

        bt_bh = backtest(np.ones_like(test_ret), test_ret, cost_bps=0)
        m_bh = calc_all_metrics(bt_bh["net_returns"])

        rng = np.random.RandomState(42)
        bt_rand = backtest(rng.randint(0, 2, size=len(test_ret)), test_ret, cost_bps=10)
        m_rand = calc_all_metrics(bt_rand["net_returns"])

        results.append({
            "fold": fname, "sma_sharpe": m_sma["sharpe"], "sma_trades": bt_sma["n_trades"],
            "bh_sharpe": m_bh["sharpe"], "rand_sharpe": m_rand["sharpe"]
        })
        print(f"  {fname}: SMA={m_sma['sharpe']:+.3f} | BH={m_bh['sharpe']:+.3f} | Rand={m_rand['sharpe']:+.3f}")
    return results


def run_lstm_fold(data_loader, fold_name, train_start, train_end, test_start, test_end):
    """单折 LSTM 训练+测试"""
    print(f"\n{'='*50}")
    print(f"  {fold_name}: {train_start[:10]}~{train_end[:10]} → {test_start[:10]}~{test_end[:10]}")
    print(f"{'='*50}")

    trainer = BTCTrainer(config, feats=ALL_FEATS)
    exp_dir = trainer.run(data_loader, train_start, train_end, n_epochs=config["trainer"]["epochs"])

    # Load best model and evaluate on test set
    best_path = os.path.join(exp_dir, "best_model.pth")
    if os.path.exists(best_path):
        trainer.model.load_state_dict(torch.load(best_path, map_location="cpu"))

    # Use full test data + 30-day lookback
    lookback_start = (pd.to_datetime(test_start) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    probs, y_true, ret = trainer.predict(data_loader, lookback_start, test_end)
    if probs is None:
        return None

    # Only score on test period
    test_n = len(data_loader.df[
        (data_loader.df["datetime"] >= pd.to_datetime(test_start)) &
        (data_loader.df["datetime"] < pd.to_datetime(test_end))
    ])
    seq = config["window"]["seq_len"]
    offset = 60 - seq  # 60d lookback - 30d seq = 30
    test_samples = min(len(probs) - offset, test_n - seq)
    if test_samples < 10:
        return None
    probs_test = probs[offset:offset + test_samples]
    ret_test = ret[offset:offset + test_samples]
    y_test = y_true[offset:offset + test_samples]

    thr, m = trainer.find_best_threshold(probs_test, ret_test)

    return {
        "fold": fold_name,
        "sharpe": m["sharpe"],
        "ann_ret": m["annualized_return"],
        "trades": m["trades"],
        "total_ret": m["total_return"],
        "threshold": thr,
        "prob_mean": float(np.mean(probs_test)),
        "prob_std": float(np.std(probs_test)),
    }


def main():
    print("=" * 60)
    print("BTC LSTM — 4折滚动窗口验证（标准训练）")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    print(f"\n数据集: {len(df)} 行, {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")

    dl = BTCDataLoader(DATA_PATH)
    print(f"特征: {len(ALL_FEATS)} ({len(PRICE_FEATS)} 价格 + {len(ALL_FEATS)-len(PRICE_FEATS)} 链上)")

    # ── Baselines ──
    print("\n" + "-" * 50)
    print("基线策略")
    print("-" * 50)
    baselines = run_baselines(df)

    # ── LSTM ──
    print("\n" + "-" * 50)
    print("LSTM 标准训练")
    print("-" * 50)
    lstm_results = []
    for fold in FOLDS:
        r = run_lstm_fold(dl, *fold)
        if r:
            lstm_results.append(r)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)

    rows = []
    for br in baselines:
        row = {"Fold": br["fold"]}
        for lr in lstm_results:
            if lr and lr["fold"] == br["fold"]:
                row["LSTM Sharpe"] = f"{lr['sharpe']:+.3f}"
                row["LSTM Trades"] = lr["trades"]
                row["Prob Std"] = f"{lr['prob_std']:.2f}"
        row["SMA Sharpe"] = f"{br['sma_sharpe']:+.3f}"
        row["BH Sharpe"] = f"{br['bh_sharpe']:+.3f}"
        row["Random Sharpe"] = f"{br['rand_sharpe']:+.3f}"
        rows.append(row)

    result_df = pd.DataFrame(rows)
    print("\n")
    print(result_df.to_string(index=False))

    if lstm_results:
        lstm_sharpes = [r["sharpe"] for r in lstm_results]
        print(f"\nLSTM 平均 Sharpe: {np.mean(lstm_sharpes):+.3f} ± {np.std(lstm_sharpes):.3f}")
    else:
        print("\n⚠️ 无有效的 LSTM 结果")

    sma_sharpes = [br["sma_sharpe"] for br in baselines]
    bh_sharpes = [br["bh_sharpe"] for br in baselines]
    rand_sharpes = [br["rand_sharpe"] for br in baselines]
    print(f"SMA  平均 Sharpe: {np.mean(sma_sharpes):+.3f} ± {np.std(sma_sharpes):.3f}")
    print(f"BH   平均 Sharpe: {np.mean(bh_sharpes):+.3f} ± {np.std(bh_sharpes):.3f}")
    print(f"Rand 平均 Sharpe: {np.mean(rand_sharpes):+.3f} ± {np.std(rand_sharpes):.3f}")

    # Save
    summary = {
        "lstm": [{"fold": r["fold"], "sharpe": float(r["sharpe"])} for r in lstm_results] if lstm_results else [],
        "sma": [{"fold": br["fold"], "sharpe": float(br["sma_sharpe"])} for br in baselines],
        "bh": [{"fold": br["fold"], "sharpe": float(br["bh_sharpe"])} for br in baselines],
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    result_df.to_csv(os.path.join(RESULTS_DIR, "all_results.csv"), index=False)
    print(f"\n结果已保存至 {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
