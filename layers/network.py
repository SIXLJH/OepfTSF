import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, scale):
        super().__init__()


        self.pred_len = int(pred_len)
        self.scale = scale


        assert self.pred_len % 4 == 0, f"pred_len must be divisible by 4, got pred_len={self.pred_len}"


        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.padding_patch = padding_patch


        self.dim = self.patch_len * self.patch_len

        self.patch_num = (seq_len - self.patch_len) // self.stride + 1
        if self.padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            self.patch_num += 1
        else:
            self.padding_patch_layer = None


        self.fc1 = nn.Linear(self.patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        self.conv1 = nn.Conv1d(
            in_channels=self.patch_num,
            out_channels=self.patch_num,
            kernel_size=self.patch_len,
            stride=self.patch_len,
            groups=self.patch_num,
        )
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)


        self.fc2 = nn.Linear(self.dim, self.patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, kernel_size=1, stride=1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)


        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * self.patch_len, self.pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(self.pred_len * 2, self.pred_len)


        self.fc5 = nn.Linear(seq_len, self.pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(self.pred_len * 2)

        self.fc6 = nn.Linear(self.pred_len * 2, self.pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(self.pred_len // 2)

        self.fc7 = nn.Linear(self.pred_len // 2, self.pred_len)


        self.fc8 = nn.Linear(self.pred_len * 2, self.pred_len * 25)


        grain_out = (self.pred_len * 25) // 4
        self.fc9 = nn.Linear(self.pred_len * 2, grain_out)

    def forward(self, s, t):

        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        B, C, I = s.shape
        s = s.reshape(B * C, I)
        t = t.reshape(B * C, I)

        if self.padding_patch == "end":
            s_ = s.unsqueeze(1)
            s_ = self.padding_patch_layer(s_)
            s = s_.squeeze(1)

        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        res = self.fc2(res)
        s = s + res


        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)


        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        t = self.fc5(t)


        t = t.unsqueeze(1)
        t = self.avgpool1(t).squeeze(1)
        t = self.ln1(t)

        t = self.fc6(t)
        t = t.unsqueeze(1)
        t = self.avgpool2(t).squeeze(1)
        t = self.ln2(t)

        t = self.fc7(t)

        x = torch.cat((s, t), dim=1)
        if self.scale == "fine":
            x = self.fc8(x)
            x = x.reshape(B, C, self.pred_len * 25)
        else:
            x = self.fc9(x)
            x = x.reshape(B, C, (self.pred_len * 25) // 4)
        x = x.permute(0, 2, 1)
        return x
