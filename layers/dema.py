import torch
from torch import nn


class DEMA(nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()
        alpha_t = torch.as_tensor(alpha, dtype=torch.float32)
        beta_t = torch.as_tensor(beta, dtype=torch.float32)
        self.register_buffer("alpha", alpha_t)
        self.register_buffer("beta", beta_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() != 3:
            raise ValueError(f"DEMA expects x with shape [B,T,C], got {tuple(x.shape)}")
        B, T, C = x.shape
        if T < 2:
            raise ValueError(f"DEMA requires sequence length T>=2, got T={T}")

        a = self.alpha.to(device=x.device, dtype=x.dtype)
        b = self.beta.to(device=x.device, dtype=x.dtype)

        # clamp for stability
        a = torch.clamp(a, 1e-6, 1.0 - 1e-6)
        b = torch.clamp(b, 1e-6, 1.0 - 1e-6)

        s_prev = x[:, 0, :]            # [B, C]
        trend = x[:, 1, :] - s_prev    # b0, [B, C]

        out = [s_prev.unsqueeze(1)]
        one_minus_a = 1.0 - a
        one_minus_b = 1.0 - b

        for t in range(1, T):
            xt = x[:, t, :]
            s = a * xt + one_minus_a * (s_prev + trend)
            trend = b * (s - s_prev) + one_minus_b * trend
            s_prev = s
            out.append(s.unsqueeze(1))

        return torch.cat(out, dim=1)
