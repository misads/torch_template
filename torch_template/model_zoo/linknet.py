import torch
import torch.nn as nn

from torch.nn import functional as F
from torchvision.models import resnet, vgg16

from collections import namedtuple

class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x


class LinkNet50(nn.Module):
    def __init__(self, out_channel=3, pad=(0, 0, 0, 0)):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNet50, self).__init__()

        base = resnet.resnet50(pretrained=False)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.pad = nn.ReflectionPad2d(pad)
        self.pad_size = [int(x / 2) for x in pad]
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = Decoder(256, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(512, 256, 3, 2, 1, 1)
        self.decoder3 = Decoder(1024, 512, 3, 2, 1, 1)
        self.decoder4 = Decoder(2048, 1024, 3, 2, 1, 1)

        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True), )
        self.tp_conv2 = nn.ConvTranspose2d(32, out_channel, 2, 2, 0)
        self.lsm = nn.ReLU()  # nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Initial block
        x = self.pad(x)
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        y = self.lsm(y)  # relu
        y = y[:,
            :,
            self.pad_size[2]:(y.shape[2] - self.pad_size[2]),
            self.pad_size[3]:(y.shape[3] - self.pad_size[3])]
        return y


class LinkNet(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, out_channel=3, pad=(0, 0, 0, 0)):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNet, self).__init__()

        base = resnet.resnet18(pretrained=False)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.pad = nn.ReflectionPad2d(pad)
        self.pad_size = [int(x / 2) for x in pad]
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True), )
        self.tp_conv2 = nn.ConvTranspose2d(32, out_channel, 2, 2, 0)
        self.lsm = nn.ReLU()

    def forward(self, x):
        # Initial block
        x = self.pad(x)
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        y = self.lsm(y)  # relu
        y = y[:,
            :,
            self.pad_size[2]:(y.shape[2] - self.pad_size[2]),
            self.pad_size[3]:(y.shape[3] - self.pad_size[3])]
        return y