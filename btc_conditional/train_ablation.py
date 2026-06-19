"""消融实验：排除 today_Open_pct（x_real 置空），训练 8 折 LSTM

目的：验证 Hybrid > LSTM 的结论是否依赖于当日开盘价信息。
today_Open_pct = price_open[t] / price_close[t-1] - 1
模型在 t 日产生信号，使用 t 日开盘价意味着用到了当天早上的信息。
排除后 x_real 为空，模型只能依赖历史窗口中的 t-1 及更早信息。

注意：today_Open_pct 不在 ALL_FEATS（历史特征）中，而是通过 x_real 进入模型。
因此消融方式是 real_feats=[] 而非从 ALL_FEATS 中排除。
"""
import os, sys
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_conditional.trainer import BTCTrainer, BTCDataLoader, ALL_FEATS

DATA_PATH = "data/processed/btc_daily_full.csv"
OUT_DIR = "btc_conditional/runs_ablation"

config = {"window": {"seq_len": 30}, "model": {"dropout": 0.3}, "trainer": {"lr": 3e-4, "epochs": 15}}

FOLDS = [
    ("Fold 1", "2018-07-01", "2020-07-01"),
    ("Fold 2", "2019-07-01", "2021-07-01"),
    ("Fold 3", "2020-07-01", "2022-07-01"),
    ("Fold 4", "2021-07-01", "2023-07-01"),
    ("Fold 5", "2022-01-01", "2024-01-01"),
    ("Fold 6", "2022-07-01", "2024-07-01"),
    ("Fold 7", "2023-01-01", "2025-01-01"),
    ("Fold 8", "2023-07-01", "2025-07-01"),
]

def main():
    print("=" * 60)
    print("消融实验：排除 today_Open_pct (x_real 置空)")
    print(f"历史特征数: {len(ALL_FEATS)} (不变)")
    print(f"x_real 维度: 0 (原始 2: today_Open_pct + 占位零)")
    print(f"输出: {OUT_DIR}")
    print("=" * 60)

    for fname, tr_s, tr_e in FOLDS:
        print(f"\n{'='*50}")
        print(f"  {fname}: {tr_s} ~ {tr_e}")
        print(f"{'='*50}")

        # real_feats=[] → x_real 为空，模型无当日开盘价信息
        dl = BTCDataLoader(DATA_PATH, feats=ALL_FEATS, target_filter=0.005, real_feats=[])
        trainer = BTCTrainer(config, feats=ALL_FEATS, label_smoothing=0.1, use_focal=True, focal_gamma=2.0, n_real=0)
        trainer.run(dl, tr_s, tr_e, n_epochs=config["trainer"]["epochs"], out_dir=OUT_DIR)
        print(f"  {fname} 训练完成")

    print("\n" + "=" * 60)
    print("消融实验 8 折训练完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
