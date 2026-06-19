"""BTC LSTM 训练器 — Focal Loss + Label Smoothing + 目标过滤"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import LSTMModel
from .backtest import backtest
from .metrics import calc_all_metrics
from .visualize import RealTimePlotter, plot_final

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRICE_FEATS = ["return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
               "volatility_10d", "volatility_20d", "volatility_30d", "vol_ratio_10_30",
               "sma_ratio_10_50", "sma_ratio_20_50",
               "rsi_14", "bb_pct_b",
               "volume_ratio", "volume_trend",
               "price_position_20d", "price_position_50d",
               "hl_spread_ma", "intraday_vol", "close_to_high"]
ONCHAIN_FEATS = ["log_tx_daily", "tx_ma_7", "fee_per_tx",
                 "hashrate_log", "difficulty_log", "nupl",
                 "active_addr_zscore", "realised_cap_log",
                 "cdd_log", "total_supply_log",
                 "mempool_size", "exchange_volume_usd",
                 "average_block_size", "fear_greed_value"]
ALL_FEATS = PRICE_FEATS + ONCHAIN_FEATS
PRICE_ONLY_FEATS = PRICE_FEATS


class FocalLoss(nn.Module):
    """Focal Loss: 让模型专注于难分类样本，忽略容易的
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    gamma=2: 标准 Focal Loss，对易分类样本降权
    alpha=0.25: 正样本权重（平衡正负样本）
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


class BCESmoothFocalLoss(nn.Module):
    """BCE + Label Smoothing + Focal Loss 融合
    
    先对标签做 label smoothing，再应用 Focal Loss
    """
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Label smoothing
        targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


class BTCDataLoader:
    def __init__(self, csv_path, seq_len=30, feats=None, target_filter=0.0, exclude_feats=None, real_feats=None):
        """target_filter: 只训练 |return| > threshold 的"有意义"日期, 0.0=不过滤
           exclude_feats: 从 feats 中排除的特征列表（用于消融实验）
           real_feats: x_real 使用的列名列表，None 则默认 ["today_Open_pct"]，空列表则无 x_real"""
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, parse_dates=["datetime"])
        self.df = self.df.sort_values("datetime").reset_index(drop=True)
        self.seq_len = seq_len
        self.feats = feats or ALL_FEATS
        if exclude_feats:
            self.feats = [f for f in self.feats if f not in exclude_feats]
        self.target_filter = target_filter
        self.df["prev_close"] = self.df["price_close"].shift(1)
        self.df["today_Open_pct"] = (self.df["price_open"] / self.df["prev_close"]) - 1
        self.df["close_to_open_return"] = (self.df["price_close"] / self.df["price_open"]) - 1
        self.df["return_1d_orig"] = self.df["price_close"].pct_change().fillna(0)
        self.df["dummy_zero"] = 0.0
        # 目标过滤标记：只有 |return| > threshold 的才用于训练
        self.df["_trainable"] = self.df["return_1d_orig"].abs() >= target_filter
        self.df["dummy_zero"] = 0.0
        # x_real 特征列
        if real_feats is None:
            self.real_feats = ["today_Open_pct", "dummy_zero"]
        else:
            self.real_feats = list(real_feats)
        # 保存原始特征值（归一化的不变源）
        self._raw_feat_values = self.df[self.feats].values.astype(np.float32).copy()
        self.feat_mean = None
        self.feat_std = None

    def _normalize(self):
        """用全样本统计量归一化（旧接口，保留兼容）"""
        vals = self._raw_feat_values.copy()
        self.feat_mean = np.nanmean(vals, axis=0)
        self.feat_std = np.nanstd(vals, axis=0) + 1e-8
        self.df[self.feats] = (vals - self.feat_mean) / self.feat_std

    def normalize_on_window(self, train_start, train_end):
        """用训练窗口统计量归一化全量数据（始终从原始值出发）"""
        mask = (self.df['datetime'] >= pd.to_datetime(train_start)) & \
               (self.df['datetime'] < pd.to_datetime(train_end))
        train_vals = self._raw_feat_values[mask.values]
        self.feat_mean = np.nanmean(train_vals, axis=0)
        self.feat_std = np.nanstd(train_vals, axis=0) + 1e-8
        self.df[self.feats] = (self._raw_feat_values - self.feat_mean) / self.feat_std

    def apply_saved_norm(self, feat_mean, feat_std):
        """用已保存的训练窗口统计量归一化（评估时使用）"""
        self.feat_mean = np.array(feat_mean, dtype=np.float32)
        self.feat_std = np.array(feat_std, dtype=np.float32)
        self.df[self.feats] = (self._raw_feat_values - self.feat_mean) / self.feat_std

    def get_window_data(self, start_date, end_date, filter_training=True):
        """filter_training=True: 训练时过滤 |return| < threshold 的噪声日"""
        mask = (self.df["datetime"] >= pd.to_datetime(start_date)) & \
               (self.df["datetime"] < pd.to_datetime(end_date))
        sub = self.df[mask].reset_index(drop=True)
        if len(sub) < self.seq_len + 1:
            return None, None, None, None
        x_hist, x_real, y, ret = [], [], [], []
        for i in range(self.seq_len, len(sub)):
            # 训练时跳过噪声日
            if filter_training and self.target_filter > 0 and not sub["_trainable"].iloc[i]:
                continue
            hist = sub[self.feats].iloc[i - self.seq_len:i].values.astype(np.float32)
            x_hist.append(hist)
            x_real.append([sub[col].iloc[i] for col in self.real_feats])
            y.append(sub["target"].iloc[i])
            ret.append(sub["return_1d_orig"].iloc[i])
        return (np.array(x_hist), np.array(x_real, dtype=np.float32),
                np.array(y, dtype=np.float32).reshape(-1, 1),
                np.array(ret, dtype=np.float32))


class BTCTrainer:
    def __init__(self, config, feats=None, label_smoothing=0.1, use_focal=True, focal_gamma=2.0, n_real=2):
        self.config = config
        self.seq_len = config["window"]["seq_len"]
        self.n_feats = len(feats or ALL_FEATS)
        self.model = LSTMModel(self.n_feats, dropout=config["model"]["dropout"], n_real=n_real).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["trainer"]["lr"])
        if use_focal:
            self.criterion = BCESmoothFocalLoss(gamma=focal_gamma, alpha=0.25, label_smoothing=label_smoothing)
        else:
            self.criterion = nn.BCELoss()
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.plotter = RealTimePlotter()
        self.epoch_losses = []
        self.epoch_accs = []
        self.epoch_val_results = []
        self.best_val_sharpe = -999

    def train_epoch(self, data_loader, train_start, train_end):
        x_hist, x_real, y, ret = data_loader.get_window_data(train_start, train_end)
        if x_hist is None:
            return 0.0, 0.0
        self.model.train()
        self.model.temperature = 1.0
        total_loss = 0.0
        eps = self.label_smoothing
        for i in range(len(x_hist)):
            xh = torch.tensor(x_hist[i:i+1], dtype=torch.float32).to(device)
            xr = torch.tensor(x_real[i:i+1], dtype=torch.float32).to(device)
            yt = torch.tensor(y[i:i+1], dtype=torch.float32).to(device)
            self.optimizer.zero_grad()
            out = self.model(xh, xr)
            if self.use_focal:
                # BCESmoothFocalLoss 内部已处理 label smoothing
                loss = self.criterion(out, yt)
            else:
                # BCELoss 需要手动 label smoothing
                yt_smooth = yt * (1 - eps) + 0.5 * eps
                loss = self.criterion(out, yt_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += float(loss)
            del xh, xr, yt, out, loss
        return total_loss / max(1, len(x_hist))

    def predict(self, data_loader, start_date, end_date, mc_dropout=False, n_mc=10):
        """批量预测并返回概率+收益+标签
        
        mc_dropout: 是否使用 MC Dropout（多次前向传播取平均，减少随机性）
        n_mc: MC Dropout 次数
        """
        x_hist, x_real, y, ret = data_loader.get_window_data(start_date, end_date, filter_training=False)
        if x_hist is None:
            return None, None, None
        
        # 推理时使用固定温度 2.0（与 runs_v4 一致）
        self.model.temperature = 2.0
        
        if mc_dropout:
            # MC Dropout: 开启 dropout，多次前向传播取平均
            self.model.train()  # 保持 dropout 开启
            all_probs = []
            for _ in range(n_mc):
                probs_i = []
                with torch.no_grad():
                    for i in range(0, len(x_hist), 16):
                        batch_end = min(i + 16, len(x_hist))
                        xh = torch.tensor(x_hist[i:batch_end], dtype=torch.float32).to(device)
                        xr = torch.tensor(x_real[i:batch_end], dtype=torch.float32).to(device)
                        out = self.model(xh, xr)
                        probs_i.extend(out[:, 0].cpu().numpy().tolist())
                all_probs.append(np.array(probs_i))
            # 取平均，减少随机性
            probs = np.mean(all_probs, axis=0)
        else:
            self.model.eval()
            probs = []
            with torch.no_grad():
                for i in range(0, len(x_hist), 16):
                    batch_end = min(i + 16, len(x_hist))
                    xh = torch.tensor(x_hist[i:batch_end], dtype=torch.float32).to(device)
                    xr = torch.tensor(x_real[i:batch_end], dtype=torch.float32).to(device)
                    out = self.model(xh, xr)
                    probs.extend(out[:, 0].cpu().numpy().tolist())
            probs = np.array(probs)
        
        return probs, y[:len(probs)], ret[:len(probs)]

    def find_best_threshold(self, probs, ret, cost_bps=10):
        """
        胜率驱动的阈值选择 — 每天看概率，只在高胜率区间买入

        核心逻辑：
        1. 遍历每个概率阈值，计算该阈值下的实际胜率
        2. 只保留胜率 > 50% 的阈值（模型预测必须优于随机）
        3. 在胜率合格的阈值中，选 Sharpe 最高的
        4. 如果没有任何阈值胜率 > 50%，回退到最优 Sharpe
        """
        if probs is None or len(probs) < 10:
            return 0.5, {"sharpe": 0.0, "trades": 0, "win_rate": 0.0}

        best_sharpe = -999
        best_thr = 0.5
        best_bt = None
        best_wr = 0.0

        # 策略1: 胜率筛选 — 只买胜率 > 50% 的概率区间
        for thr in np.arange(0.30, 0.81, 0.01):
            sigs = (probs >= thr).astype(int)
            bt = backtest(sigs, ret[:len(sigs)], cost_bps=cost_bps)
            if bt["n_trades"] < 5:
                continue
            
            # 计算实际胜率（交易日的收益 > 0 的比例）
            nr = bt["net_returns"]
            trade_mask = nr != 0
            if trade_mask.sum() < 5:
                continue
            win_rate = float((nr[trade_mask] > 0).mean())
            
            # 胜率必须 > 50% 才考虑
            if win_rate <= 0.50:
                continue
            
            m = calc_all_metrics(nr)
            if m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_thr = thr
                best_bt = bt
                best_wr = win_rate

        # 策略2: 如果没有胜率 > 50% 的阈值，回退到最优 Sharpe（不限胜率）
        if best_bt is None:
            for thr in np.arange(0.20, 0.81, 0.01):
                sigs = (probs >= thr).astype(int)
                bt = backtest(sigs, ret[:len(sigs)], cost_bps=cost_bps)
                if bt["n_trades"] < 5:
                    continue
                nr = bt["net_returns"]
                trade_mask = nr != 0
                if trade_mask.sum() < 5:
                    continue
                win_rate = float((nr[trade_mask] > 0).mean())
                m = calc_all_metrics(nr)
                if m["sharpe"] > best_sharpe:
                    best_sharpe = m["sharpe"]
                    best_thr = thr
                    best_bt = bt
                    best_wr = win_rate

        # 策略3: 终极回退
        if best_bt is None:
            thr = float(np.median(probs))
            sigs = (probs >= thr).astype(int)
            bt = backtest(sigs, ret[:len(sigs)], cost_bps=cost_bps)
            nr = bt["net_returns"]
            trade_mask = nr != 0
            best_wr = float((nr[trade_mask] > 0).mean()) if trade_mask.sum() > 0 else 0.0
            best_thr = thr
            best_bt = bt

        m = calc_all_metrics(best_bt["net_returns"])
        return best_thr, {**m, "trades": best_bt["n_trades"], "win_rate": best_wr}

    def run(self, data_loader, train_start, train_end, n_epochs=30, out_dir="btc_conditional/runs", exp_name=None):
        data_loader.normalize_on_window(train_start, train_end)
        # 支持自定义实验名称（用于滚动窗口）
        if exp_name:
            exp_dir = os.path.join(out_dir, exp_name)
        else:
            exp_dir = os.path.join(out_dir, f"train_{train_start[:10]}")
        os.makedirs(exp_dir, exist_ok=True)
        best_model_path = os.path.join(exp_dir, "best_model.pth")
        self.epoch_losses = []
        self.epoch_accs = []
        self.epoch_val_results = []
        self.best_val_sharpe = -999
        self.best_val_threshold = None  # 保存最佳验证集阈值

        # Use last 20% of training data as validation
        df = data_loader.df
        tr_mask = (df["datetime"] >= pd.to_datetime(train_start)) & \
                  (df["datetime"] < pd.to_datetime(train_end))
        train_dates = df[tr_mask]["datetime"].values
        split_idx = int(len(train_dates) * 0.8)
        if split_idx < 50:
            val_start = train_start
            val_end = train_end
        else:
            val_start = pd.to_datetime(train_dates[split_idx]).strftime("%Y-%m-%d")
            val_end = train_end

        val_probs, val_y, val_ret = None, None, None
        for epoch in range(n_epochs):
            t0 = time.time()
            avg_loss = self.train_epoch(data_loader, train_start, train_end)
            epoch_time = time.time() - t0

            # Validation (every 5 epochs or last epoch)
            if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
                val_probs, val_y, val_ret = self.predict(data_loader, val_start, val_end)
                val_thr, val_m = self.find_best_threshold(val_probs, val_ret)
                print(f"  Epoch {epoch+1:2d}/{n_epochs} | Loss {avg_loss:.4f} | "
                      f"Val Sharpe {val_m['sharpe']:+.3f} | Trades {val_m['trades']} | "
                      f"WinRate {val_m['win_rate']:.1%} | Thr {val_thr:.2f}", flush=True)
                self.epoch_val_results.append({
                    "sharpe": val_m["sharpe"], "threshold": val_thr, "trades": val_m["trades"]
                })
                if val_m["sharpe"] > self.best_val_sharpe:
                    self.best_val_sharpe = val_m["sharpe"]
                    self.best_val_threshold = val_thr  # 保存最佳阈值
                    torch.save(self.model.state_dict(), best_model_path)
            else:
                print(f"  Epoch {epoch+1:2d}/{n_epochs} | Loss {avg_loss:.4f}", flush=True)

            self.epoch_losses.append(avg_loss)

        # 保存阈值 + 归一化统计量到文件
        if self.best_val_threshold is not None:
            import json
            threshold_path = os.path.join(exp_dir, "threshold.json")
            save_data = {
                "threshold": self.best_val_threshold,
                "temperature": 2.0,  # 固定温度
                "val_sharpe": self.best_val_sharpe,
                "train_start": train_start,
                "train_end": train_end,
                # 归一化统计量（评估时必须使用，确保训练/推理预处理一致）
                "feat_mean": data_loader.feat_mean.tolist(),
                "feat_std": data_loader.feat_std.tolist(),
            }
            with open(threshold_path, "w") as f:
                json.dump(save_data, f, indent=2)

        return exp_dir
