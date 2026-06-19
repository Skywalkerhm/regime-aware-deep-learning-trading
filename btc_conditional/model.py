import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """两层 LSTM + 实时特征融合 + 可校准温度缩放"""
    def __init__(self, n_features, hidden1=128, hidden2=64, dropout=0.3, n_real=2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2 + n_real, 1)
        self.sigmoid = nn.Sigmoid()
        # 温度参数：训练时=1.0，推理时通过校准学习最优值
        self.temperature = 1.0

    def forward(self, x_hist, x_real):
        """x_hist: (batch, seq_len, n_features), x_real: (batch, 2)"""
        out, _ = self.lstm1(x_hist)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = out[:, -1, :]
        combined = torch.cat([out, x_real], dim=1)
        logits = self.fc(combined)
        return self.sigmoid(logits / self.temperature)
