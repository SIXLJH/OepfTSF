import torch
from torch import nn


class EMA(nn.Module):


    def __init__(self, alpha):
        super().__init__()

        alpha_t = torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() != 3:
            raise ValueError(f"EMA expects x with shape [B,T,C], got {tuple(x.shape)}")


        a = self.alpha.to(device=x.device, dtype=x.dtype)


        a = torch.clamp(a, 1e-6, 1.0 - 1e-6)

        B, T, C = x.shape

        s = x[:, 0, :]
        out = [s.unsqueeze(1)]
        one_minus = 1.0 - a
        for t in range(1, T):
            xt = x[:, t, :]
            s = a * xt + one_minus * s
            out.append(s.unsqueeze(1))

        return torch.cat(out, dim=1)
