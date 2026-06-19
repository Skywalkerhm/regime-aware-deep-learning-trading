import pandas as pd
import numpy as np
import os
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("BTC 数据管线 v1 — 合并 Kaggle + Yahoo 构建统一日频数据集")
print("=" * 60)

# ── 1. Yahoo Finance OHLCV (2014-09 ~ 2026-06) ──
print("\n[1/4] 加载 Yahoo Finance 价格...")
yahoo = pd.read_csv(RAW / "yahoo_price.csv", parse_dates=["date"])
yahoo = yahoo.sort_values("date").reset_index(drop=True)
yahoo = yahoo.rename(columns={
    "date": "datetime",
    "close": "price_close",
    "open": "price_open",
    "high": "price_high",
    "low": "price_low",
    "volume": "price_volume",
})
yahoo = yahoo[["datetime", "price_open", "price_high", "price_low", "price_close", "price_volume"]]
print(f"   {len(yahoo)} 行, {yahoo['datetime'].min().date()} ~ {yahoo['datetime'].max().date()}")

# ── 2. blockchain.com 链上日数据 (2009~2023) ──
print("\n[2/4] 加载 blockchain.com 链上数据...")
bc = pd.read_csv(RAW / "blockchain_dot_com_daily_data.csv", parse_dates=["datetime"])
bc = bc.sort_values("datetime").reset_index(drop=True)
bc_feat = ["datetime", "mempool_size", "transaction_rate", "average_block_size",
           "exchange_volume_usd", "average_confirmation_time", "hash_rate",
           "difficulty", "miners_revenue", "total_transaction_fees"]
bc = bc[bc_feat]
print(f"   {len(bc)} 行, {bc['datetime'].min().date()} ~ {bc['datetime'].max().date()}")

# ── 3. lookintobitcoin 数据 (2010~2023) ──
print("\n[3/4] 加载 lookintobitcoin 数据...")
lib = pd.read_csv(RAW / "look_into_bitcoin_daily_data.csv", parse_dates=["datetime"])
lib = lib.sort_values("datetime").reset_index(drop=True)
lib_feat = ["datetime", "total_supply", "realised_cap_usd", "nupl",
            "coin_days_destroyed", "active_addresses", "fear_greed_value",
            "lightning_nodes", "lightning_capacity_usd"]
lib = lib[lib_feat]
print(f"   {len(lib)} 行, {lib['datetime'].min().date()} ~ {lib['datetime'].max().date()}")

# ── 4. 合并 ──
print("\n[4/4] 合并数据集...")
df = yahoo.merge(bc, on="datetime", how="left")
df = df.merge(lib, on="datetime", how="left")
print(f"   合并后: {len(df)} 行, {len(df.columns)} 列")

# ── 5. 特征工程 ──
print("\n   计算特征...")

# 价格收益率
df["return_1d"] = df["price_close"].pct_change()
df["return_3d"] = df["price_close"].pct_change(3)
df["return_5d"] = df["price_close"].pct_change(5)
df["return_10d"] = df["price_close"].pct_change(10)
df["return_20d"] = df["price_close"].pct_change(20)

# 多尺度波动率
df["volatility_10d"] = df["return_1d"].rolling(10).std()
df["volatility_20d"] = df["return_1d"].rolling(20).std()
df["volatility_30d"] = df["return_1d"].rolling(30).std()
df["vol_ratio_10_30"] = df["volatility_10d"] / (df["volatility_30d"] + 1e-8)

# 均线
df["sma_10"] = df["price_close"].rolling(10).mean()
df["sma_20"] = df["price_close"].rolling(20).mean()
df["sma_50"] = df["price_close"].rolling(50).mean()
df["sma_ratio_10_50"] = df["sma_10"] / df["sma_50"]
df["sma_ratio_20_50"] = df["sma_20"] / df["sma_50"]

# RSI 14
delta = df["price_close"].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / (avg_loss + 1e-8)
df["rsi_14"] = 100 - (100 / (1 + rs))

# Bollinger Bands %B
df["bb_mid"] = df["price_close"].rolling(20).mean()
df["bb_std"] = df["price_close"].rolling(20).std()
df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
df["bb_pct_b"] = (df["price_close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-8)

# 成交量特征
df["volume_ratio"] = df["price_volume"] / (df["price_volume"].rolling(20).mean() + 1e-8)
df["volume_trend"] = df["price_volume"].rolling(5).mean() / (df["price_volume"].rolling(20).mean() + 1e-8)

# 价格动量 / 位置
df["price_position_20d"] = (df["price_close"] - df["price_close"].rolling(20).min()) / \
    (df["price_close"].rolling(20).max() - df["price_close"].rolling(20).min() + 1e-8)
df["price_position_50d"] = (df["price_close"] - df["price_close"].rolling(50).min()) / \
    (df["price_close"].rolling(50).max() - df["price_close"].rolling(50).min() + 1e-8)

# High-Low spread
df["hl_spread"] = (df["price_high"] / (df["price_low"] + 1e-8)) - 1
df["hl_spread_ma"] = df["hl_spread"].rolling(10).mean()

# 日内波动 vs 日间波动
df["intraday_vol"] = (df["price_high"] / df["price_low"] - 1).rolling(10).mean()
df["close_to_high"] = df["price_close"] / (df["price_high"] + 1e-8)
df["close_to_open"] = df["price_close"] / (df["price_open"] + 1e-8)

# 链上活动特征
df["log_tx_daily"] = np.log1p(df["transaction_rate"])
df["tx_ma_7"] = df["transaction_rate"].rolling(7).mean()
df["fee_per_tx"] = df["total_transaction_fees"] / (df["transaction_rate"] + 1e-8)
df["hashrate_log"] = np.log1p(df["hash_rate"])
df["difficulty_log"] = np.log1p(df["difficulty"])

# 链上估值特征
df["active_addr_zscore"] = (
    df["active_addresses"] - df["active_addresses"].rolling(30).mean()
) / (df["active_addresses"].rolling(30).std() + 1e-8)
df["realised_cap_log"] = np.log1p(df["realised_cap_usd"])
df["cdd_log"] = np.log1p(df["coin_days_destroyed"])
df["total_supply_log"] = np.log1p(df["total_supply"])
df["nupl"] = df["nupl"]  # already a ratio [-1,1]

# ── 6. 前向填充链上特征 (2023-09 后空缺) ──
onchain_cols = ["mempool_size", "transaction_rate", "hash_rate", "difficulty",
                "active_addresses", "nupl", "realised_cap_usd", "coin_days_destroyed",
                "total_supply", "fear_greed_value", "log_tx_daily", "tx_ma_7",
                "fee_per_tx", "hashrate_log", "difficulty_log", "active_addr_zscore",
                "realised_cap_log", "cdd_log", "total_supply_log",
                "exchange_volume_usd", "average_block_size",
                "average_confirmation_time", "miners_revenue", "total_transaction_fees",
                "lightning_nodes", "lightning_capacity_usd"]
available = [c for c in onchain_cols if c in df.columns]
df[available] = df[available].ffill()
print(f"   前向填充 {len(available)} 个链上特征")

# ── 7. 标签: 下日涨跌二分类 ──
df["target"] = np.sign(df["return_1d"].shift(-1))
# 日频上 return_1d 几乎不会精确为 0，但保险起见映射 -1/0 → -1, 1 → 1
df["target"] = df["target"].map({-1.0: 0, 0.0: 0, 1.0: 1})

# ── 8. 清理缺失值 (开头 rolling 产生的 NaN) ──
before = len(df)
df = df.dropna(subset=["target"]).reset_index(drop=True)
print(f"\n   去除标签 NaN: {before} → {len(df)} 行")

# ── 9. 保存 ──
out_path = OUT / "btc_daily_full.csv"
df.to_csv(out_path, index=False)
print(f"\n{'=' * 60}")
print(f"Done: {out_path}")
print(f"   行数: {len(df)}, 列数: {len(df.columns)}")
print(f"   时间: {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")
print(f"   特征: {', '.join(df.columns[:15])}...")
print(f"   标签分布: {df['target'].value_counts().to_dict()}")
print(f"{'=' * 60}")
