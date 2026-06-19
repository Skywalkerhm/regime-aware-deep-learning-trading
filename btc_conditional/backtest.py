import numpy as np

def backtest(signals, returns, cost_bps=10, direction="long"):
    """
    回测函数 - 修复信号与收益的时间对齐问题
    
    信号在 t 时刻生成，实际执行在 t+1 时刻，收益使用 t+1 时刻的收益
    交易成本在执行日扣除
    """
    signals = np.asarray(signals)
    returns = np.asarray(returns)
    cost = cost_bps / 10000.0
    prev_pos = 0
    net_returns = np.zeros_like(returns)
    n_trades = 0
    
    # 从第2天开始，使用前一天的信号
    for t in range(1, len(signals)):
        # 使用前一天的信号作为今天的仓位
        pos = 1 if signals[t-1] >= 0.5 else 0
        
        if pos != prev_pos:
            n_trades += 1
            # 新仓位在 t 时刻执行，收益为 t 时刻的收益，成本在执行日扣除
            net_returns[t] = pos * returns[t] - cost
        else:
            net_returns[t] = pos * returns[t]
        
        prev_pos = pos
    
    # 第一天没有前一天的信号，收益为0
    net_returns[0] = 0
    
    return {"net_returns": net_returns, "n_trades": n_trades, "gross_returns": returns}
