"""训练全部 8 折 LSTM — Focal Loss + 目标过滤 + 多尺度特征"""
import os, sys
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_conditional.trainer import BTCTrainer, BTCDataLoader, ALL_FEATS

DATA_PATH = "data/processed/btc_daily_full.csv"
OUT_DIR = "btc_conditional/runs_v7"  # 温度校准（无 LayerNorm）

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
    print("训练全部 8 折 (LayerNorm + 温度校准 + Focal Loss)")
    print(f"特征数: {len(ALL_FEATS)}")
    print("=" * 60)

    for fname, tr_s, tr_e in FOLDS:
        print(f"\n{'='*50}")
        print(f"  {fname}: {tr_s} ~ {tr_e}")
        print(f"{'='*50}")

        # target_filter=0.005: 只训练 |return| > 0.5% 的有意义日期
        dl = BTCDataLoader(DATA_PATH, feats=ALL_FEATS, target_filter=0.005)
        trainer = BTCTrainer(config, feats=ALL_FEATS, label_smoothing=0.1, use_focal=True, focal_gamma=2.0)
        trainer.run(dl, tr_s, tr_e, n_epochs=config["trainer"]["epochs"], out_dir=OUT_DIR)
        print(f"  {fname} 训练完成")

    print("\n" + "=" * 60)
    print("全部 8 折训练完成")
    print("=" * 60)

if __name__ == "__main__":
    main()