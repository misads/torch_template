# encoding=utf-8
"""Misc PyTorch utils

Usage:
    >>> from torch_template import torch_utils
    >>> torch_utils.func_name()  # to call functions in this file

"""
from datetime import datetime
import math
import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np

##############################
#    Functional utils
##############################
from misc_utils import format_num


def clamp(x, min=0.01, max=0.99):
    """clamp a tensor.

    Args:
        x(torch.Tensor): input tensor
        min(float): value < min will be set to min.
        max(float): value > max will be set to max.

    Returns:
        (torch.Tensor): a clamped tensor.
    """
    return torch.clamp(x, min, max)


def repeat(x: torch.Tensor, *sizes):
    """Repeat a dimension of a tensor.

    Args:
        x(torch.Tensor): input tensor.
        sizes: repeat times for each dimension.

    Returns:
        (torch.Tensor): a repeated tensor.

    Example
        >>> t = repeat(t, 1, 3, 1, 1)  # same as t = t.repeat(1, 3, 1, 1) or t = torch.cat([t, t, t], dim=1)
    """

    return x.repeat(*sizes)


def tensor2im(x: torch.Tensor, norm=False, to_save=False):
    """Convert tensor to image.

    Args:
        x(torch.Tensor): input tensor, [n, c, h, w] float32 type.
        norm(bool): if the tensor should be denormed first
        to_save(bool): if False, a float32 image of [h, w, c], if True, a uint8 image of [h, w, c].

    Returns:
        an image in shape of [h, w, c] if to_save else [c, h, w].

    """
    if norm:
        x = (x + 1) / 2
    x[x > 1] = 1
    x[x < 0] = 0

    x = x.detach().cpu().data[0]

    if to_save:
        x *= 255
        x = x.astype(np.uint8)
        x = x.transpose((1, 2, 0))

    return x


##############################
#    Network utils
##############################
def print_network(net: nn.Module, print_size=False):
    """Print network structure and number of parameters.

    Args:
        net(nn.Module): network model.
        print_size(bool): print parameter num of each layer.

    Example
        >>> import torchvision as tv
        >>> from torch_template import torch_utils
        >>>
        >>> vgg16 = tv.models.vgg16()
        >>> torch_utils.print_network(vgg16)
        >>> '''
        >>> features.0.weight [3, 64, 3, 3]
        >>> features.2.weight [64, 64, 3, 3]
        >>> features.5.weight [64, 128, 3, 3]
        >>> features.7.weight [128, 128, 3, 3]
        >>> features.10.weight [128, 256, 3, 3]
        >>> features.12.weight [256, 256, 3, 3]
        >>> features.14.weight [256, 256, 3, 3]
        >>> features.17.weight [256, 512, 3, 3]
        >>> features.19.weight [512, 512, 3, 3]
        >>> features.21.weight [512, 512, 3, 3]
        >>> features.24.weight [512, 512, 3, 3]
        >>> features.26.weight [512, 512, 3, 3]
        >>> features.28.weight [512, 512, 3, 3]
        >>> classifier.0.weight [25088, 4096]
        >>> classifier.3.weight [4096, 4096]
        >>> classifier.6.weight [4096, 1000]
        >>> Total number of parameters: 138,357,544
        >>> '''
    """
    num_params = 0
    print(net)
    for name, param in net.named_parameters():
        num_params += param.numel()
        size = list(param.size())
        if len(size) > 1:
            if print_size:
                print(name, size[1:2]+size[:1]+size[2:], format_num(param.numel()))
            else:
                print(name, size[1:2] + size[:1] + size[2:])
    print('Total number of parameters: %s' % format_num(num_params))
    # print('The size of receptive field: %s' % format_num(receptive_field(net)))


# def receptive_field(net):
#     def _f(output_size, ksize, stride, dilation):
#         return (output_size - 1) * stride + ksize * dilation - dilation + 1
#
#     stats = []
#     for m in net.modules():
#         if isinstance(m, torch.nn.Conv2d):
#             stats.append((m.kernel_size, m.stride, m.dilation))
#
#     rsize = 1
#     for (ksize, stride, dilation) in reversed(stats):
#         if type(ksize) == tuple: ksize = ksize[0]
#         if type(stride) == tuple: stride = stride[0]
#         if type(dilation) == tuple: dilation = dilation[0]
#         rsize = _f(rsize, ksize, stride, dilation)
#     return rsize


##############################
#    Abstract Meters class
##############################


class Meters(object):
    def __init__(self):
        pass

    def update(self, new_dic):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def items(self):
        return self.dic.items()


class AverageMeters(Meters):
    """AverageMeter class

    Example
        >>> avg_meters = AverageMeters()
        >>> for i in range(100):
        >>>     avg_meters.update({'f': i})
        >>>     print(str(avg_meters))

    """

    def __init__(self, dic=None, total_num=None):
        self.dic = dic or {}
        # self.total_num = total_num
        self.total_num = total_num or {}

    def update(self, new_dic):
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = new_dic[key]
                self.total_num[key] = 1
            else:
                self.dic[key] += new_dic[key]
                self.total_num[key] += 1
        # self.total_num += 1

    def __getitem__(self, key):
        return self.dic[key] / self.total_num[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self):
        return self.dic.keys()


class ExponentialMovingAverage(Meters):
    """EMA class

    Example
        >>> ema_meters = ExponentialMovingAverage(0.98)
        >>> for i in range(100):
        >>>     ema_meters.update({'f': i})
        >>>     print(str(ema_meters))

    """

    def __init__(self, decay=0.9, dic=None, total_num=None):
        self.decay = decay
        self.dic = dic or {}
        # self.total_num = total_num
        self.total_num = total_num or {}

    def update(self, new_dic):
        decay = self.decay
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = (1 - decay) * new_dic[key]
                self.total_num[key] = 1
            else:
                self.dic[key] = decay * self.dic[key] + (1 - decay) * new_dic[key]
                self.total_num[key] += 1
        # self.total_num += 1

    def __getitem__(self, key):
        return self.dic[key]  # / self.total_num[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self):
        return self.dic.keys()

##############################
#    Checkpoint helper
##############################


def load_ckpt(model, ckpt_path):
    """Load checkpoint.

    Args:
        model(nn.Module): object of a subclass of nn.Module.
        ckpt_path(str): *.pt file to load.

    Example
        >>> class Model(nn.Module):
        >>>     pass
        >>>
        >>> model = Model().cuda()
        >>> load_ckpt(model, 'model.pt')

    """
    model.load_state_dict(torch.load(ckpt_path))


def save_ckpt(model, ckpt_path):
    """Save checkpoint.

    Args:
        model(nn.Module): object of a subclass of nn.Module.
        ckpt_path(str): *.pt file to save.

    Example
        >>> class Model(nn.Module):
        >>>     pass
        >>>
        >>> model = Model().cuda()
        >>> save_ckpt(model, 'model.pt')

    """
    torch.save(model.state_dict(), ckpt_path)


##############################
#    LR_Scheduler
##############################


class LR_Scheduler(object):
    """Learning Rate Scheduler

    Example:
        >>> scheduler = LR_Scheduler('cos', opt.lr, opt.epochs, len(dataloader), warmup_epochs=20)
        >>> for i, data in enumerate(dataloader)
        >>>     scheduler(self.g_optimizer, i, epoch)

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``  每到达lr_step, lr就乘以0.1

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0, logger=None):
        """
            :param mode: `step` `cos` or `poly`
            :param base_lr:
            :param num_epochs:
            :param iters_per_epoch:
            :param lr_step: lr step to change lr/ for `step` mode
            :param warmup_epochs:
            :param logger:
        """
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.logger = logger
        if logger:
            self.logger.info('Using {} LR Scheduler!'.format(self.mode))

    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            if self.logger:
                self.logger.info('\n=>Epoches %i, learning rate = %.4f' % (epoch, lr))
            else:
                print('\nepoch: %d lr: %.6f' % (epoch, lr))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10

"""
    TensorBoard
    Example:
        writer = create_summary_writer(os.path.join(self.basedir, 'logs'))
        write_meters_loss(writer, 'train', avg_meters, iteration)
        write_loss(writer, 'train', 'F1', 0.78, iteration)
        write_image(writer, 'train', 'input', img, iteration)
        # shell
        tensorboard --logdir {base_path}/logs

"""


def create_summary_writer(log_dir):
    """Create a tensorboard summary writer.

    Args:
        log_dir: log directory.

    Returns:
        (SummaryWriter): a summary writer.

    Example
        >>> writer = create_summary_writer(os.path.join(self.basedir, 'logs'))
        >>> write_meters_loss(writer, 'train', avg_meters, iteration)
        >>> write_loss(writer, 'train', 'F1', 0.78, iteration)
        >>> write_image(writer, 'train', 'input', img, iteration)
        >>> # shell
        >>> tensorboard --logdir {base_path}/logs

    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, datetime.now().strftime('%m-%d_%H-%M-%S'))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir, max_queue=3, flush_secs=10)

    return writer


def write_loss(writer: SummaryWriter, prefix, loss_name: str, value: float, iteration):
    """Write loss into writer.

    Args:
        writer(SummaryWriter): writer created by create_summary_writer()
        prefix(str): any string, e.g. 'train'.
        loss_name(str): loss name.
        value(float): loss value.
        iteration(int): epochs or iterations.

    Example
        >>> write_loss(writer, 'train', 'F1', 0.78, iteration)

    """
    writer.add_scalar(
        os.path.join(prefix, loss_name), value, iteration)


def write_graph(writer: SummaryWriter, model, inputs_to_model=None):
    """Write net graph into writer.

    Args:
        writer(SummaryWriter): writer created by create_summary_writer()
        model(nn.Module): model.
        inputs_to_model(tuple or list): forward inputs.

    Example
        >>> from tensorboardX import SummaryWriter
        >>> input_data = Variable(torch.rand(16, 3, 224, 224))
        >>> vgg16 = torchvision.models.vgg16()
        >>>
        >>> writer = SummaryWriter(log_dir='logs')
        >>> write_graph(vgg16, (input_data,))

    """
    with writer:
        writer.add_graph(model, inputs_to_model)


def write_image(writer: SummaryWriter, prefix, image_name: str, img, iteration, dataformats='CHW'):
    """Write images into writer.

    Args:
        writer(SummaryWriter): writer created by create_summary_writer()
        prefix(str): any string, e.g. 'train'.
        image_name(str): image name.
        img: image tensor in [C, H, W] shape.
        iteration(int): epochs or iterations.
        dataformats(str): 'CHW' or 'HWC' or 'NCHW'.

    Example
        >>> write_image(writer, 'train', 'input', img, iteration)

    """
    writer.add_image(
        os.path.join(prefix, image_name), img, iteration, dataformats=dataformats)


def write_meters_loss(writer: SummaryWriter, prefix, avg_meters: Meters, iteration):
    """Write all losses in a meter class into writer.

    Args:
        writer(SummaryWriter): writer created by create_summary_writer()
        prefix(str): any string, e.g. 'train'.
        avg_meters(AverageMeters or ExponentialMovingAverage): meters.
        iteration(int): epochs or iterations.

    Example
        >>> writer = create_summary_writer(os.path.join(self.basedir, 'logs'))
        >>> ema_meters = ExponentialMovingAverage(0.98)
        >>> for i in range(100):
        >>>     ema_meters.update({'f1': i, 'f2': i*0.5})
        >>>     write_meters_loss(writer, 'train', ema_meters, i)

    """
    for key in avg_meters.keys():
        meter = avg_meters[key]
        writer.add_scalar(
            os.path.join(prefix, key), meter, iteration)


