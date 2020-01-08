import pdb

import numpy as np
import torch
import os

from torch import optim
import torch.nn.functional as F


from torch_template import model_zoo
from torch_template.loss.seg_loss import bce_loss, dice_loss, BCEFocalLoss

from torch_template.network.base_model import BaseModel
from torch_template.network.metrics import ssim, L1_loss
from torch_template.utils.torch_utils import ExponentialMovingAverage, print_network


models = {
    'Nested': model_zoo['NestedUNet']()

}


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.cleaner = models[opt.model].cuda(device=opt.device)
        #####################
        #    Init weights
        #####################
        self.cleaner.apply(weights_init)

        print_network(self.cleaner)

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

    def update_G(self, img_var, y):
        opt = self.opt

        # cleaned = x
        cleaned = self.cleaner(img_var)

        #########################
        #       sigmoid
        #########################

        # cleaned = cleaned.mean(dim=1, keepdim=True)
        # y = y.mean(dim=1, keepdim=True)
        # f1 = f1_loss(cleaned, y, thresh=160/255)

        prediction = F.sigmoid(cleaned)
        target = F.sigmoid(y)

        #########################
        #       losses
        #########################
        bce = bce_loss(prediction, target) * opt.weight_bce

        dice = dice_loss(prediction, target) * opt.weight_dice
        l1 = L1_loss(prediction, target)

        # pdb.set_trace()
        loss = bce + dice

        # GAN loss
        # loss_gen_adv = self.discriminitor.calc_gen_loss(input_fake=cleaned)
        self.avg_meters.update({'bce': bce.item(), 'dice': dice.item(), 'l1': l1.item()})

        #loss_gen = loss + loss_gen_adv * 1.
        self.g_optimizer.zero_grad()
        loss.backward()
        self.g_optimizer.step()

        return cleaned

    def update_D(self, x, y):
        self.d_optimizer.zero_grad()
        # encode
        cleaned = self.cleaner(x)
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
