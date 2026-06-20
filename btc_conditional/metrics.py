import numpy as np

def calc_all_metrics(returns):
    """
    计算策略的各项指标
    
    使用 252 个交易日进行年化计算（金融市场标准）
    """
    if len(returns) < 2:
        return {"sharpe": 0.0, "annualized_return": 0.0,
                "annualized_vol": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "total_return": 0.0}
    
    # 使用 252 个交易日进行年化（金融市场标准）
    trading_days = 252
    
    ann_ret = float(np.mean(returns) * trading_days)
    ann_vol = float(np.std(returns) * np.sqrt(trading_days))
    sharpe = ann_ret / (ann_vol + 1e-8)
    
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / running_max
    max_dd = float(np.min(drawdown))
    
    win_rate = float(np.mean(returns > 0))
    total_ret = float(np.prod(1 + returns) - 1)
    
    return {"sharpe": sharpe, "annualized_return": ann_ret,
            "annualized_vol": ann_vol, "max_drawdown": max_dd,
            "win_rate": win_rate, "total_return": total_ret}


def compute_calibration(probs, y_true, n_bins=10):
    """ECE and Brier score for probability calibration assessment."""
    import numpy as np
    from sklearn.metrics import brier_score_loss
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for i in range(n_bins):
        m = bin_idx == i
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(probs)) * abs(y_true[m].mean() - probs[m].mean())
    brier = brier_score_loss(y_true, probs)
    return {"ece": float(ece), "brier": float(brier), "n": len(probs)}