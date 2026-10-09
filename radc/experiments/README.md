# Experiments

计划中的实验入口：

- `synthetic/`：无信息 regime、高分离 regime、低概率分散、弱检测器等合成场景。
- `btc/`：BTC LSTM、Transformer、XGBoost 的 nested walk-forward。
- `tsla/`：TSLA LSTM 外部确认。
- `evaluate_candidate.py`：统一候选评价入口。
- `run_alpha_evolve.py`：AlphaEvolve 搜索循环入口。

实验状态规则：

- search、validation、external confirmation 必须使用不同 fold；
- 每次评价保存 JSON、运行时间、stdout/stderr 和配置哈希；
- 失败候选也必须归档；
- 任何表格和图必须能追溯到候选 ID、数据包哈希和命令。
