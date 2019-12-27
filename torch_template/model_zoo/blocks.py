import torch
import torch.nn as nn
from torch.nn import functional as F


class PALayer(nn.Module):
    def __init__(self, channel=64):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel=64):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class ConvLayer(torch.nn.Sequential):
    """
        conv=nn.Conv2d
        norm=nn.BatchNorm2d
        ConvLayer(conv, in_channels, n_feats, kernel_size, stride=1, norm=None, act=act)

    """

    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None,
                 act=None):
        super(ConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, dropRate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        out = x + x2
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out
