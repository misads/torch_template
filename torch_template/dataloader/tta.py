# encoding=utf-8
"""
    TTA plugin used in test data_loader loop, containing overlap and data_aug (8×)

    Author: zks@tju.edu.cn

    Refactor: xuhaoyu@tju.edu.cn
"""

import pdb

import torch
from torch.autograd import Variable
from torchvision.transforms import transforms


class OverlapTTA(object):
    """overlap TTA

    Args:
        nw(int): num of patches (in width direction)
        nh(int):  num of patches (in height direction)
        patch_w(int): width of a patch.
        patch_h(int): height of a patch.
        norm_patch(bool): if norm each patch or not.
        flip_aug(bool): not used yet.
        device(str): device string, default 'cuda:0'.

    Usage Example
        >>> from torch_template import OverlapTTA
        >>> for i, data in enumerate(dataset):
        >>>     tta = OverlapTTA(img, 10, 10, 256, 256, norm_patch=False, flip_aug=False, device=opt.device)
        >>>     for j, x in enumerate(tta):  # 获取每个patch输入
        >>>         generated = model(x)
        >>>         torch.cuda.empty_cache()
        >>>         tta.collect(generated[0], j)  # 收集inference结果
        >>>     output = tta.combine()

    """
    def __init__(self, img, nw, nh, patch_w=256, patch_h=256, norm_patch=False, flip_aug=False, device='cuda:0'):

        self.img = img
        self.nw = nw
        self.nh = nh
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.N, self.C, self.H, self.W = img.shape
        self.norm_patch = norm_patch
        self.flip_aug = flip_aug
        self.transforms = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5),
                                                                   (0.5, 0.5, 0.5))]) if norm_patch else None
        self.device = device

        #####################################
        #                 步长
        #####################################
        stride_h = (self.H - 256) // (nh - 1)
        stride_w = (self.W - 256) // (nw - 1)

        self.overlap_times = torch.zeros((self.C, self.H, self.W)).cpu()
        self.slice_h = []
        self.slice_w = []

        #####################################
        #   除了最后一个patch, 都按照固定步长取块
        # 将位置信息先保存在slice_h和slice_w数组中
        #####################################
        for i in range(nh - 1):
            self.slice_h.append([i * stride_h, i * stride_h + 256])
        self.slice_h.append([self.H - 256, self.H])
        for i in range(nw - 1):
            self.slice_w.append([i * stride_w, i * stride_w + 256])
        self.slice_w.append([self.W - 256, self.W])

        #####################################
        #             保存结果的数组
        #####################################
        self.result = torch.zeros((self.C, self.H, self.W)).cpu()

    def collect(self, x, cur):
        x = x.detach().cpu()

        j = cur % self.nw
        i = (cur - j) // self.nh

        #####################################
        #         分别记录图像和重复次数
        #####################################
        self.result[:, self.slice_h[i][0]:self.slice_h[i][1], self.slice_w[j][0]:self.slice_w[j][1]] += x
        self.overlap_times[:, self.slice_h[i][0]:self.slice_h[i][1], self.slice_w[j][0]:self.slice_w[j][1]] += 1

    def combine(self):
        if self.flip_aug:
            pass
        else:
            return self.result / self.overlap_times

    def __getitem__(self, index):
        """
            获取tta patch作为网络输入
            :param index:
            :return:
        """
        if self.flip_aug:
            pass

        else:
            j = index % self.nh
            i = index // self.nw
            img = self.img[:, :, self.slice_h[i][0]:self.slice_h[i][1], self.slice_w[j][0]:self.slice_w[j][1]]
            if self.norm_patch:
                img = self.transforms(img[0]).unsqueeze(dim=0)

            img_var = Variable(img, requires_grad=False).cuda(device=self.device)
            return img_var

    def __len__(self):
        return self.nw * self.nh
