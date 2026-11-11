import torch
from torch import nn

from layers.ema import EMA
from layers.dema import DEMA


class DECOMP(nn.Module):

    def __init__(self, ma_type, alpha, beta=None):
        super().__init__()
        self.ma_type = ma_type

        if ma_type == "ema":
            self.ma = EMA(alpha)
        elif ma_type == "dema":
            if beta is None:
                raise ValueError("DECOMP with ma_type='dema' requires beta")
            self.ma = DEMA(alpha, beta)
        elif ma_type == "reg":
            self.ma = None
        else:
            raise ValueError(f"Unknown ma_type='{ma_type}'. Expected one of: 'ema', 'dema', 'reg'")

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        if self.ma_type == "reg" or self.ma is None:
            # no decomposition
            return x, x

        trend = self.ma(x)
        seasonal = x - trend
        return seasonal, trend

