import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.decomp import DECOMP
from layers.network import Network
from layers.revin import RevIN



def _as_tensor_25(x):
    t = torch.as_tensor(x, dtype=torch.float32).view(-1)
    if t.numel() != 25:
        raise ValueError(f"Support points must have 25 elements, got {t.numel()}")
    return t


def _default_interleaved_supports(n=25, lo=-4.0, hi=4.0):
    idx_1 = torch.linspace(lo, hi, n, dtype=torch.float32)
    step = (hi - lo) / (n - 1)
    idx_2 = idx_1 + 0.5 * step
    return idx_1, idx_2


def _entropy_confidence(p, eps=1e-8):
    p = torch.clamp(p, min=eps)
    H = -torch.sum(p * torch.log(p), dim=2)
    return 1.0 / (H + eps)


def support_expectation(
    p,
    idx,
    trim_ratio=0.1,
    temperature=0.7,
    eps=1e-8
):

    logp = torch.log(p + eps) / temperature
    p = torch.softmax(logp, dim=2)

    if trim_ratio > 0:
        k = int(25 * trim_ratio)
        _, order = torch.sort(p, dim=2)
        mask = torch.ones_like(p)
        mask.scatter_(2, order[:, :, :k, :], 0.0)
        mask.scatter_(2, order[:, :, -k:, :], 0.0)
        p = p * mask
        p = p / (p.sum(dim=2, keepdim=True) + eps)

    return torch.einsum("bhkc,ki->bhic", p, idx).squeeze(2)


def _soft_label_from_support(y, support, sigma, eps=1e-8):
    s = support.view(1, 1, 25, 1).to(y.device, y.dtype)
    y_ = y.unsqueeze(2)
    dist2 = (y_ - s) ** 2
    q = torch.softmax(-dist2 / (2 * sigma * sigma), dim=2)
    return q / torch.clamp(q.sum(dim=2, keepdim=True), min=eps)


def _make_horizon_weights(H, device, dtype):
    h = torch.arange(1, H + 1, device=device, dtype=dtype).view(H, 1)
    w = (-torch.atan(h) + math.pi / 4 + 1)
    return w / (w.mean() + 1e-8)


def support_mean(p, idx):
    return torch.einsum("bhkc,ki->bhic", p, idx).squeeze(2)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        seq_len = configs.seq_len
        pred_len = configs.pred_len
        c_in = configs.enc_in
        assert pred_len % 4 == 0

        self.revin = bool(getattr(configs, "revin", False))
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        self.ma_type = configs.ma_type
        self.decomp = None
        if self.ma_type != "reg":
            self.decomp = DECOMP(configs.ma_type, configs.alpha, configs.beta)

        self.net_1 = Network(seq_len, pred_len, configs.patch_len, configs.stride, configs.padding_patch, "fine")
        self.net_2 = Network(seq_len, pred_len, configs.patch_len, configs.stride, configs.padding_patch, "fine")
        self.net_3 = Network(seq_len, pred_len, configs.patch_len, configs.stride, configs.padding_patch, "grain")
        self.net_4 = Network(seq_len, pred_len, configs.patch_len, configs.stride, configs.padding_patch, "grain")

        idx1, idx2 = _default_interleaved_supports(25, -4.0, 4.0)
        self.register_buffer("idx_1", idx1.view(25, 1))
        self.register_buffer("idx_2", idx2.view(25, 1))

        self.dist_sup_weight = configs.dist_sup_weight
        self.label_sigma_ratio = configs.label_sigma_ratio
        self.support_reg_weight = configs.support_reg_weight
        self.support_min_gap_ratio = configs.support_min_gap_ratio
        self.support_smooth_weight = configs.support_smooth_weight
        self.grad_clip = configs.grad_clip
        self.eps = 1e-8

    def forward(self, x, y=None):
        if self.revin:
            x = self.revin_layer(x, "norm")

        if self.decomp is not None:
            s, t = self.decomp(x)
        else:
            s, t = x, x

        x1 = self.net_1(s, t)
        x2 = self.net_2(s, t)
        x3 = self.net_3(s, t)
        x4 = self.net_4(s, t)

        B, L25, C = x1.shape
        H = L25 // 25

        x1 = x1.view(B, H, 25, C)
        x2 = x2.view(B, H, 25, C)

        p1 = torch.softmax(x1, dim=2)
        p2 = torch.softmax(x2, dim=2)

        out1 = support_mean(p1, self.idx_1)
        out2 = support_mean(p2, self.idx_2)

        conf1 = _entropy_confidence(p1) ** 0.85
        conf2 = _entropy_confidence(p2) ** 0.85


        dis = (out1 - out2).abs().detach()
        conf1 = conf1 / (1.0 + 0.1 * dis)
        conf2 = conf2 / (1.0 + 0.1 * dis)

        mask_raw = conf1 / (conf1 + conf2 + self.eps)


        h = torch.arange(1, H + 1, device=mask_raw.device, dtype=mask_raw.dtype).view(1, H, 1)

        sharp = 1.0 + 0.6 * torch.exp(-h / 24.0)
        mask = mask_raw ** sharp
        mask = mask / (mask + (1.0 - mask_raw) ** sharp + self.eps)

        output_fine = out1 * mask + out2 * (1 - mask)

        output_down = output_fine.view(B, H // 4, 4, C).mean(dim=2)

        x3 = x3.view(B, H // 4, 25, C)
        x4 = x4.view(B, H // 4, 25, C)

        p3 = torch.softmax(x3, dim=2)
        p4 = torch.softmax(x4, dim=2)

        out3 = support_mean(p3, self.idx_1)
        out4 = support_mean(p4, self.idx_2)

        conf3 = _entropy_confidence(p3)
        conf4 = _entropy_confidence(p4)
        mask_g = conf3 / (conf3 + conf4 + self.eps)

        output_grain = out3 * mask_g + out4 * (1 - mask_g)

        if self.revin:
            output = self.revin_layer(output_fine, "denorm")
        else:
            output = output_fine

        con1 = F.mse_loss(out1, out2)
        con2 = F.mse_loss(out3, out4)
        con3 = F.mse_loss(output_grain, output_down)


        if y is not None:
            if self.revin:
                y = self.revin_layer(y, "norm")

            step = (4.0 - (-4.0)) / 24.0
            sigma = step * self.label_sigma_ratio

            q = _soft_label_from_support(y, self.idx_1, sigma)
            p = torch.clamp(
                p1 * mask.unsqueeze(2) + p2 * (1 - mask).unsqueeze(2),
                min=self.eps
            )

            ce = -(q * torch.log(p)).sum(dim=2)
            con3 = con3 + self.dist_sup_weight * ce.mean()

        return output, con1, con2, con3
