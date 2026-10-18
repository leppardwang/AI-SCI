from __future__ import annotations

import torch
from torch import nn


class RiskLSTM(nn.Module):
    """本地边缘 LSTM 风险推理模型（torch CPU 可训可推理）。

    结构：LSTM(input→hidden, num_layers) → 取最后时刻隐状态 → Linear → 单 logit。
    对应论文第 5 章"边缘节点轻量风险推理"设计，训练与推理全本地完成。
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bias=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)
