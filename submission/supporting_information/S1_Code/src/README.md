# Source Layout

- `rutl/policy.py`：AlphaEvolve 可编辑的唯一核心文件，定义 RUTL 阈值策略接口。
- `rutl/data.py`：Phase 1 实现 canonical prediction bundle 读取与 schema 校验。
- `rutl/metrics.py`：Phase 1 实现净效用、回撤、换手和 block bootstrap。
- `rutl/backtest.py`：Phase 1 实现与旧代码解耦的 long/flat 回测。
- `rutl/evaluator.py`：Phase 1 实现 JSON evaluator 和 cascade。

当前只建立策略接口；数据和评价代码必须在 G1 前完成并测试。
