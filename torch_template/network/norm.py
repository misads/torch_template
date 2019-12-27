import torch

import torch.nn as nn


class InsNorm(nn.Module):
    def __init__(self, dim, eps=1e-9):
        super(InsNorm, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        flat_len = x.size(2) * x.size(3)
        vec = x.view(x.size(0), x.size(1), flat_len)
        mean = torch.mean(vec, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        var = torch.var(vec, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((flat_len - 1) / float(flat_len))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out
