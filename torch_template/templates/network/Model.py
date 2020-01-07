import pdb

import numpy as np
import torch
import os

from torch import optim

import network.DuRN_Pure_Conv_3_Refactor as pure
import model_zoo
# from network.pyramid_ppp import Pyramid_Net
from network.Ms_Discriminator import MsImageDis
from network.base_model import BaseModel
import network.DuRN_Pure_Conv_3_coarse_3 as coarse_3
from network.metrics import ssim, L1_loss
from network.weights_init import init_weights
from utils.torch_utils import ExponentialMovingAverage, print_network

models = {
    'default': pure.cleaner(),
    'pure': pure.cleaner(),
    'coarse': coarse_3.cleaner(),
    'FFA': model_zoo.FFA(),
    'Nested': model_zoo.NestedUNet(),
}


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()

        # self.cleaner = Pyramid_Net(3, 256).cuda(device=opt.device)
        self.cleaner = models[opt.model].cuda(device=opt.device)
        if opt.init:
            init_weights(self.cleaner, opt.init)

        print_network(self.cleaner)

        params = {
            'dim': 64,
            'norm': 'none',
            'activ': 'lrelu',  # activation function [relu/lrelu/prelu/selu/tanh]
            'n_layer': 4,  # number of layers in D
            'gan_type': 'lsgan',  # GAN loss [lsgan/nsgan]
            'num_scales': 3,  # number of scales
            'pad_type': 'reflect'
        }

        # self.discriminitor = MsImageDis(input_dim=3, params=params).cuda(device=opt.device)
        #
        # print(self.discriminitor)

        self.g_optimizer = optim.Adam(self.cleaner.parameters(), lr=opt.lr)
        # self.d_optimizer = optim.Adam(cleaner.parameters(), lr=opt.lr)

        # load networks
        if opt.load:
            pretrained_path = opt.load
            self.load_network(self.cleaner, 'G', opt.which_epoch, pretrained_path)
            # if self.training:
            #     self.load_network(self.discriminitor, 'D', opt.which_epoch, pretrained_path)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

    def update_G(self, x, y):

        # L1 & SSIM loss
        some_loss = 0
        cleaned = self.cleaner(x)
        ssim_loss_r = -ssim(cleaned, y)
        ssim_loss = ssim_loss_r * 1.1

        # Compute L1 loss (not used)
        l1_loss = L1_loss(cleaned, y)
        l1_loss = l1_loss * 0.75

        loss = ssim_loss + l1_loss

        # GAN loss
        # loss_gen_adv = self.discriminitor.calc_gen_loss(input_fake=cleaned)
        self.avg_meters.update({'ssim': -ssim_loss_r.item(), 'L1': l1_loss.item()})

        #loss_gen = loss + loss_gen_adv * 1.
        self.g_optimizer.zero_grad()
        loss.backward()
        self.g_optimizer.step()

        return cleaned

    def update_D(self, x, y):
        self.d_optimizer.zero_grad()
        # encode
        cleaned = cleaner(x)
        # h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)

        # D loss
        loss_dis = self.discriminitor.calc_dis_loss(input_fake=cleaned, input_real=y)
        self.avg_meters.update({'dis': loss_dis})

        loss_dis = loss_dis * 1.  # weights
        loss_dis.backward()
        self.d_optimizer.step()
        return cleaned

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, x):
        return self.cleaner(x)

    def inference(self, x, image=None):
        pass

    def save(self, which_epoch):
        self.save_network(self.cleaner, 'G', which_epoch)
        # self.save_network(self.discriminitor, 'D', which_epoch)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
