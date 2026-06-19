"""滚动窗口训练 — 每 6 个月重新训练，使用 2 年历史数据"""
import os, sys
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_conditional.trainer import BTCTrainer, BTCDataLoader, ALL_FEATS

DATA_PATH = "data/processed/btc_daily_full.csv"
OUT_DIR = "btc_conditional/runs_v7"  # 温度校准（无 LayerNorm）

config = {"window": {"seq_len": 30}, "model": {"dropout": 0.3}, "trainer": {"lr": 3e-4, "epochs": 15}}

# 滚动窗口：每 6 个月重新训练，使用 2 年历史
# 格式: (训练开始, 训练结束, 测试开始, 测试结束, 名称)
ROLLING_WINDOWS = [
    # 2020-07 ~ 2021-01 (训练: 2018-07 ~ 2020-07)
    ("2018-07-01", "2020-07-01", "2020-07-01", "2021-01-01", "roll_2020-07"),
    # 2021-01 ~ 2021-07 (训练: 2019-01 ~ 2021-01)
    ("2019-01-01", "2021-01-01", "2021-01-01", "2021-07-01", "roll_2021-01"),
    # 2021-07 ~ 2022-01 (训练: 2019-07 ~ 2021-07)
    ("2019-07-01", "2021-07-01", "2021-07-01", "2022-01-01", "roll_2021-07"),
    # 2022-01 ~ 2022-07 (训练: 2020-01 ~ 2022-01)
    ("2020-01-01", "2022-01-01", "2022-01-01", "2022-07-01", "roll_2022-01"),
    # 2022-07 ~ 2023-01 (训练: 2020-07 ~ 2022-07)
    ("2020-07-01", "2022-07-01", "2022-07-01", "2023-01-01", "roll_2022-07"),
    # 2023-01 ~ 2023-07 (训练: 2021-01 ~ 2023-01)
    ("2021-01-01", "2023-01-01", "2023-01-01", "2023-07-01", "roll_2023-01"),
    # 2023-07 ~ 2024-01 (训练: 2021-07 ~ 2023-07)
    ("2021-07-01", "2023-07-01", "2023-07-01", "2024-01-01", "roll_2023-07"),
    # 2024-01 ~ 2024-07 (训练: 2022-01 ~ 2024-01)
    ("2022-01-01", "2024-01-01", "2024-01-01", "2024-07-01", "roll_2024-01"),
    # 2024-07 ~ 2025-01 (训练: 2022-07 ~ 2024-07)
    ("2022-07-01", "2024-07-01", "2024-07-01", "2025-01-01", "roll_2024-07"),
    # 2025-01 ~ 2025-07 (训练: 2023-01 ~ 2025-01)
    ("2023-01-01", "2025-01-01", "2025-01-01", "2025-07-01", "roll_2025-01"),
    # 2025-07 ~ 2026-01 (训练: 2023-07 ~ 2025-07)
    ("2023-07-01", "2025-07-01", "2025-07-01", "2026-01-01", "roll_2025-07"),
]

def main():
    print("=" * 60)
    print("滚动窗口训练 (2 年历史, 每 6 个月重新训练)")
    print(f"特征数: {len(ALL_FEATS)}")
    print(f"共 {len(ROLLING_WINDOWS)} 个窗口")
    print("=" * 60)

    for tr_s, tr_e, te_s, te_e, name in ROLLING_WINDOWS:
        print(f"\n{'='*50}")
        print(f"  {name}: 训练 {tr_s}~{tr_e} -> 测试 {te_s}~{te_e}")
        print(f"{'='*50}")

        dl = BTCDataLoader(DATA_PATH, feats=ALL_FEATS, target_filter=0.005)
        trainer = BTCTrainer(config, feats=ALL_FEATS, label_smoothing=0.1, use_focal=True, focal_gamma=2.0)
        trainer.run(dl, tr_s, tr_e, n_epochs=config["trainer"]["epochs"], out_dir=OUT_DIR, exp_name=name)
        print(f"  {name} 训练完成")

    print("\n" + "=" * 60)
    print("全部滚动窗口训练完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
