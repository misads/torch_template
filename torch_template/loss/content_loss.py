from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet, vgg16

L1_loss = nn.L1Loss()
L2_loss = nn.MSELoss()


class VGG19Loss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGG19Loss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class VGG16Loss(nn.Module):
    def __init__(self, device):
        super(VGG16Loss, self).__init__()
        self.vgg = Vgg16(requires_grad=False).to(device)
        self.loss = nn.MSELoss()

    def forward(self, output, target, weight=1):
        f_output = self.vgg(output).relu2_2
        f_target = self.vgg(target).relu2_2
        return weight * self.loss(f_output, f_target)


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):  # 1: relu1_1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):  # 6: relu2_1
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):  # 11: relu3_1
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):  # 20: relu4_2
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):  # 29: relu5_3
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1_1 = self.slice1(X)
        h_relu2_1 = self.slice2(h_relu1_1)
        h_relu3_1 = self.slice3(h_relu2_1)
        h_relu4_2 = self.slice4(h_relu3_1)
        h_relu5_3 = self.slice5(h_relu4_2)
        out = [h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_2, h_relu5_3]
        return out


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        """
            Vgg16：
                return: relu1_2, relu2_2, relu3_3, relu4_3
            :param requires_grad:
        """
        super(Vgg16, self).__init__()
        vgg_pretrained_features = vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):  # 3: relu1_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):  # 8: relu2_2
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):  # 15: relu3_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):  # 22: relu4_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
