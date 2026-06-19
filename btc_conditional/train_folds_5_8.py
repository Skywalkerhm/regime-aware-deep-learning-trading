"""训练 Fold 5-8 的 LSTM 模型（补充缺失的 checkpoint）"""
import os, sys
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_conditional.trainer import BTCTrainer, BTCDataLoader, ALL_FEATS

DATA_PATH = "data/processed/btc_daily_full.csv"
OUT_DIR = "btc_conditional/runs_v2"

config = {"window": {"seq_len": 30}, "model": {"dropout": 0.3}, "trainer": {"lr": 3e-4, "epochs": 15}}

# Fold 5-8 的训练/测试区间
FOLDS = [
    ("Fold 5", "2022-01-01", "2024-01-01"),
    ("Fold 6", "2022-07-01", "2024-07-01"),
    ("Fold 7", "2023-01-01", "2025-01-01"),
    ("Fold 8", "2023-07-01", "2025-07-01"),
]

def main():
    print("=" * 60)
    print("训练 Fold 5-8 LSTM 模型")
    print("=" * 60)

    for fname, tr_s, tr_e in FOLDS:
        ckpt_path = os.path.join(OUT_DIR, f"train_{tr_s}", "best_model.pth")
        if os.path.exists(ckpt_path):
            print(f"\n{fname} ({tr_s}~{tr_e}): 已存在 checkpoint，跳过")
            continue

        print(f"\n{'='*50}")
        print(f"  {fname}: {tr_s} ~ {tr_e}")
        print(f"{'='*50}")

        dl = BTCDataLoader(DATA_PATH, feats=ALL_FEATS)
        trainer = BTCTrainer(config, feats=ALL_FEATS)
        trainer.run(dl, tr_s, tr_e, n_epochs=config["trainer"]["epochs"], out_dir=OUT_DIR)
        print(f"  {fname} 训练完成")

    print("\n" + "=" * 60)
    print("全部 Fold 5-8 训练完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
