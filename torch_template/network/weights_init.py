import torch.nn as nn
import numpy as np
from torch.nn import init


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Sequential):
        return
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.orthogonal(m.weight.data, gain=1)
    elif isinstance(m, nn.Linear):
        init.orthogonal(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('[i] initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'edsr':
        pass
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)