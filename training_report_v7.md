# LSTM + Regime Hybrid 策略 — 训练方法与实验结果说明

> 版本：v7 | 日期：2026-06-19 | 数据区间：2014-09 ~ 2026-01

---

## 1. 数据预处理

### 1.1 数据来源

| 来源 | 内容 | 时间范围 |
|------|------|----------|
| Yahoo Finance | BTC-USD OHLCV 日频数据 | 2014-09 ~ 2026-06 |
| blockchain.com | 链上指标（hash_rate, difficulty, active_addresses 等） | 2009 ~ 2023 |
| lookintobitcoin | NUPL、 realised_cap、fear_greed 等 | 2010 ~ 2023 |

合并脚本：`build_dataset.py`，输出：`data/processed/btc_daily_full.csv`

### 1.2 特征工程（20 个价格特征 + 14 个链上特征 = 34 维）

**价格/技术特征（ALL_FEATS 中的 PRICE_FEATS）：**

| 类别 | 特征 |
|------|------|
| 多尺度收益率 | return_1d, return_3d, return_5d, return_10d, return_20d |
| 多尺度波动率 | volatility_10d, volatility_20d, volatility_30d, vol_ratio_10_30 |
| 均线比率 | sma_ratio_10_50, sma_ratio_20_50 |
| 技术指标 | rsi_14, bb_pct_b |
| 成交量 | volume_ratio, volume_trend |
| 价格位置 | price_position_20d, price_position_50d |
| 日内波动 | hl_spread_ma, intraday_vol, close_to_high |

**链上特征（ONCHAIN_FEATS）：**

log_tx_daily, tx_ma_7, fee_per_tx, hashrate_log, difficulty_log, nupl, active_addr_zscore, realised_cap_log, cdd_log, total_supply_log, mempool_size, exchange_volume_usd, average_block_size, fear_greed_value

### 1.3 标签定义

二分类标签：`target = sign(return_1d.shift(-1))`，即预测次日涨跌方向。

---

## 2. 模型架构

```
输入: x_hist (batch, seq_len=30, n_features=34) + x_real (batch, 2)
  │
  ├─ LSTM1(34 → 128) → Dropout(0.3)
  ├─ LSTM2(128 → 64) → Dropout(0.3)
  ├─ 取最后时间步 → cat(x_real)  → (batch, 66)
  └─ Linear(66 → 1) → Sigmoid(logits / temperature)
  
temperature = 2.0（温度缩放，使概率输出更平滑）
```

模型文件：`btc_conditional/model.py`

---

## 3. 训练配置

| 参数 | 值 |
|------|-----|
| 序列长度 (seq_len) | 30 天 |
| 隐藏层 | 128 → 64 |
| Dropout | 0.3 |
| 学习率 | 3e-4 |
| 训练轮次 | 15 epochs |
| 损失函数 | BCE + Label Smoothing(0.1) + Focal Loss(γ=2, α=0.25) |
| 温度缩放 | sigmoid(logits / 2.0) |
| 目标过滤 | 仅训练 \|return\| > 0.5% 的"有意义"日期 |
| 归一化 | 仅用训练窗口统计量（z-score），避免数据泄露 |
| 阈值选择 | 三级回退：绝对阈值 → 百分位 → 中位数 |

训练器：`btc_conditional/trainer.py`

---

## 4. Walk-Forward 8-Fold 交叉验证设计

采用时间序列 walk-forward 方案，每折训练 2 年、测试 6 个月，严格保证时间顺序：

| Fold | 训练期 | 测试期 | 市场阶段 |
|------|--------|--------|----------|
| 1 | 2018-07 ~ 2020-07 | 2020-07 ~ 2021-01 | 复苏 (Recovery) |
| 2 | 2019-07 ~ 2021-07 | 2021-07 ~ 2022-01 | 高峰 (Peak) |
| 3 | 2020-07 ~ 2022-07 | 2022-07 ~ 2023-01 | 崩盘 (Crash) |
| 4 | 2021-07 ~ 2023-07 | 2023-07 ~ 2024-01 | 复苏 (Recovery) |
| 5 | 2022-01 ~ 2024-01 | 2024-01 ~ 2024-07 | 牛市 (Bull) |
| 6 | 2022-07 ~ 2024-07 | 2024-07 ~ 2025-01 | 牛市 (Bull) |
| 7 | 2023-01 ~ 2025-01 | 2025-01 ~ 2025-07 | 牛市 (Bull) |
| 8 | 2023-07 ~ 2025-07 | 2025-07 ~ 2026-01 | 熊市 (Bear) |

模型 checkpoint 保存在 `btc_conditional/runs_v7/train_{date}/` 下，每折包含 `best_model.pth` 和 `threshold.json`。

---

## 5. 五策略对比设计

### 5.1 策略定义

| 策略 | 说明 |
|------|------|
| **Buy & Hold** | 始终持有，基准策略 |
| **SMA(20)** | 收盘价 > 20日均线时持有，否则空仓 |
| **Regime (纯规则)** | 检测市场状态为 trend_up 时买入，否则空仓 |
| **LSTM (纯模型)** | 模型概率 ≥ 固定阈值时买入 |
| **Hybrid (LSTM+Regime)** | LSTM 概率 + 状态依赖阈值（核心策略） |

### 5.2 Hybrid 策略机制 — 状态依赖阈值

**市场状态检测规则：**

```
若 volatility_20d > vol_p75×2 或 return_20d < -10%  →  crisis
若 |sma_ratio_20_50 - 1| > 0.025                     →  trend_up / trend_down
若 |sma_ratio_20_50 - 1| > 0.008                     →  trend_up / trend_down
其他                                                   →  chop
```

**状态依赖阈值：**

| 市场状态 | 阈值 | 含义 |
|----------|------|------|
| trend_up | 0.40 | 趋势向上时降低门槛，更积极买入 |
| trend_down | 0.50 | 下行趋势中保持中性 |
| chop | 0.55 | 震荡市提高门槛，减少假信号 |
| crisis | 0.80 | 危机时极高门槛，几乎不交易 |

### 5.3 回测细节

- **滞后机制**：`signals[t-1] × returns[t]`，确保 t 日信号在 t+1 日执行，杜绝未来数据泄露
- **交易成本**：10 bps 单边手续费
- **Sharpe 计算**：252 日年化（金融市场标准）
- **累计收益**：`cumprod(1 + net_returns) - 1`

---

## 6. 实验结果

### 6.1 各 Fold 五策略 Sharpe 对比（归一化修复后）

| Fold | 市场阶段 | Buy&Hold | SMA | Regime | LSTM | **Hybrid** | 提升 |
|------|----------|----------|-----|--------|------|------------|------|
| 1 | Recovery | +3.798 | **+3.095** | +1.254 | +2.849 | **+3.274** | +0.425 |
| 2 | Peak | +0.981 | **+0.422** | +1.346 | +2.205 | **+2.426** | +0.221 |
| 3 | Crash | -0.297 | **-1.595** | +0.613 | +0.011 | **+0.336** | +0.325 |
| 4 | Recovery | +1.574 | **+1.958** | -0.254 | -0.204 | **+1.488** | +1.693 |
| 5 | Bull | +1.425 | **+0.414** | -0.236 | +1.726 | **+2.675** | +0.949 |
| 6 | Bull | +1.478 | **+1.076** | +0.261 | -0.380 | **+1.248** | +1.627 |
| 7 | Bull | +0.682 | **+0.655** | +0.636 | +0.996 | **+0.890** | -0.106 |
| 8 | Bear | -0.778 | **+0.170** | +0.625 | -0.866 | **-0.176** | +0.691 |

### 6.2 策略排名（平均 Sharpe）

| 排名 | 策略 | 平均 Sharpe |
|------|------|-------------|
| #1 | **Hybrid (LSTM+Regime)** | **+1.520** |
| #2 | Buy & Hold | +1.108 |
| #3 | LSTM (pure) | +0.792 |
| #4 | SMA(20) | +0.775 |
| #5 | Regime (pure rule) | +0.531 |

### 6.3 Hybrid vs LSTM 统计检验

| 指标 | 值 |
|------|-----|
| Hybrid 平均 Sharpe | +1.520 ± 1.117 |
| LSTM 平均 Sharpe | +0.792 ± 1.268 |
| 平均提升 | **+0.728** |
| Paired t-test p-value | **0.0162** |
| Wilcoxon p-value | **0.0156** |
| 显著性 (p < 0.05) | **YES**（两种检验均显著） |
| Hybrid 优于 LSTM 的 Fold 数 | **7/8** |

### 6.4 各市场状态表现

| 市场状态 | Hybrid Sharpe | LSTM Sharpe | B&H Sharpe | Fold 数 |
|----------|---------------|-------------|------------|---------|
| Recovery | +2.381 | +1.322 | +2.686 | 2 |
| Peak | +2.426 | +2.205 | +0.981 | 1 |
| Crash | +0.336 | +0.011 | -0.297 | 1 |
| Bull | +1.604 | +0.781 | +1.195 | 3 |
| Bear | -0.176 | -0.866 | -0.778 | 1 |

**结论：Hybrid 在所有市场状态均优于 LSTM（7/8 Fold），尤其在 Bear 市场提升最为显著（-0.866 → -0.176）。**

---

## 7. 关键设计决策与消融

### 7.1 解决交易数量过少问题

原始 LSTM 使用固定 0.5 阈值，导致交易信号极少（多数 Fold 仅 0-3 笔）。改进措施：

1. **温度缩放** (T=2.0)：使 sigmoid 输出更平滑，概率分布更集中
2. **Focal Loss** (γ=2)：让模型关注难分类样本，提升整体预测质量
3. **Label Smoothing** (0.1)：防止模型过度自信
4. **目标过滤** (|return| > 0.5%)：仅在有意义的波动日训练
5. **状态依赖阈值**：Hybrid 策略在 trend_up 时降低阈值至 0.40，增加交易机会

### 7.2 防止数据泄露

- 归一化仅用训练窗口统计量
- 回测信号滞后 1 天：`signals[t-1] × returns[t]`
- Walk-forward 严格时间分割，无重叠

---

## 8. 文件索引

### 核心代码

| 文件 | 说明 |
|------|------|
| `build_dataset.py` | 数据管线：合并原始数据 → 特征工程 → btc_daily_full.csv |
| `btc_conditional/model.py` | LSTM 模型定义 |
| `btc_conditional/trainer.py` | 训练器（Focal Loss + 归一化 + 阈值选择） |
| `btc_conditional/backtest.py` | 回测函数（滞后信号 + 交易成本） |
| `btc_conditional/metrics.py` | 指标计算（Sharpe, 年化收益, 最大回撤） |
| `btc_conditional/validate_hybrid.py` | Hybrid 策略完整验证（8 折 + 统计检验） |
| `btc_conditional/regime_final.py` | 五策略对比主脚本 |

### 模型与结果

| 路径 | 内容 |
|------|------|
| `btc_conditional/runs_v7/` | 8 折模型 checkpoint + threshold.json |
| `btc_conditional/results_v2/` | 实验结果 CSV + summary |
| `data/processed/btc_daily_full.csv` | 处理后的完整数据集 |

---

## 9. 归一化一致性修复记录

### 问题

v7 初始版本中，`trainer.run()` 通过 `normalize_on_window()` 对训练数据做 z-score 归一化，但评估脚本（`validate_hybrid.py`、`regime_final.py`、`continuous_all.py`、`evaluate_rolling.py`、`analyze_all.py`）新建 `BTCDataLoader` 后从未调用归一化，导致 `predict()` 喂给模型的是**原始未归一化特征**（如 `realised_cap_log ≈ 27`、`mempool_size ≈ 百万级`），而模型是在 z-score 输入（均值≈0、标准差≈1）上训练的。

### 修复方案

1. **`BTCDataLoader.__init__`**：保存 `_raw_feat_values`（原始特征值的不可变副本）
2. **`normalize_on_window()`**：始终从 `_raw_feat_values` 出发计算 z-score（支持多次调用）
3. **`apply_saved_norm(feat_mean, feat_std)`**：新方法，用已保存的训练窗口统计量归一化
4. **`trainer.run()`**：训练结束后将 `feat_mean`、`feat_std` 写入 `threshold.json`
5. **所有评估脚本**：从 `threshold.json` 读取 `feat_mean`/`feat_std`，在 `predict()` 前调用 `dl.apply_saved_norm()`

### 修复前后对比

| 指标 | 修复前（未归一化） | 修复后 |
|------|-------------------|--------|
| Hybrid 平均 Sharpe | +1.871 | +1.520 |
| LSTM 平均 Sharpe | +0.951 | +0.792 |
| Hybrid-LSTM 提升 | +0.920 | +0.728 |
| p-value (t-test) | 0.039 | 0.0162 |
| Hybrid 优于 LSTM | 8/8 | 7/8 |

修复后 LSTM Sharpe 下降（证实之前是噪声），但 Hybrid 仍稳健领先，且统计显著性更强（p=0.016 vs 0.039）。

---

## 10. 设计选择说明

- **target_filter=0.005**：训练时丢弃 |return| < 0.5% 的低波动日，仅保留"有意义"的涨跌日。这改变了模型学习的标签分布（偏向更大波动），predict 时不过滤。论文中需说明此设计选择。
- **today_Open_pct 作为 x_real 输入**：决策日 t 的开盘价用于信号生成，配合 t-1 决策、t 日交易的回测机制。论文需明确成交时点假设（假设收盘价成交还是开盘价成交）。
- **x_real 第二维恒为 0**：历史遗留占位维，无实际作用，不影响模型行为。

---

## 11. 复现步骤

```bash
# 1. 数据预处理
python build_dataset.py

# 2. 训练 8 折模型（已训练完成，checkpoint 在 runs_v7/）
python btc_conditional/train_all_folds.py

# 3. 运行 Hybrid 验证
python btc_conditional/validate_hybrid.py

# 4. 查看综合结果
python btc_conditional/analyze_all.py
```
