# python 2.7, pytorch 0.3.1

import os, sys
import pdb

from dataloader.image_folder import get_data_loader_folder
from network.Model import Model
from options import opt
from utils.misc_utils import get_file_name

sys.path.insert(1, '../')
import torch
import torchvision
import numpy as np
import subprocess
import random
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image

from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ski_ssim
import utils.misc_utils as utils

# ------- Option --------
tag = 'epoch_500'
# Choose a dataset.
data_name = 'rtts'  # 'DCPDNData' or 'RESIDE'
# -----------------------

if data_name == 'RESIDE':
    testroot = "./datasets/" + data_name + "/sots_indoor_test/"
    test_list_pth = "../lists/RESIDE_indoor/sots_test_list.txt"
elif data_name == 'DCPDNData':
    testroot = "./datasets/" + data_name + "/TestA/"
    test_list_pth = '../lists/' + data_name + '/testA_list.txt'
elif data_name == 'rtts':
    testroot = "./datasets/RESIDE/" + data_name
elif data_name == 'real':
    testroot = "./datasets/RESIDE/" + data_name
else:
    print('Unknown dataset name.')


def test(cleaner, dataloader, data_name='RESIDE'):
    show_dst = os.path.join(opt.result_dir, opt.tag)
    ave_psnr = 0.0
    ave_ssim = 0.0
    ct_num = 0
    print('Start testing ' + tag + '...')
    for i, data in enumerate(dataloader):
        if data_name == 'RESIDE':
            hazy, label, im_name = data
            b, v, c, h, w = hazy.size()
            for bi in range(b):
                label_v = label[bi]
                label_v = label_v.numpy()
                label_v *= 255
                label_v = label_v.astype(np.uint8)
                label_v = label_v.transpose((1, 2, 0))

                for vi in range(v):
                    utils.progress_bar(i * v + vi, len(dataloader) * v)
                    ct_num += 1
                    hazy_vi = hazy[bi, vi]
                    hazy_vi = hazy_vi.unsqueeze(dim=0)
                    hazy_vi = Variable(hazy_vi, requires_grad=False).cuda(device=opt.device)
                    res = cleaner(hazy_vi)
                    res = res.data.cpu().numpy()[0]
                    res[res > 1] = 1
                    res[res < 0] = 0
                    res *= 255
                    res = res.astype(np.uint8)
                    res = res.transpose((1, 2, 0))
                    ave_psnr += psnr(res, label_v, data_range=255)
                    ave_ssim += ski_ssim(res, label_v, data_range=255, multichannel=True)
                    Image.fromarray(res).save(show_dst + im_name[0].split('.')[0] + '_' + str(vi + 1) + '.png')

        elif data_name == 'SINGLE':
            hazy_1, im_name = data
            hazy = Variable(hazy_1, requires_grad=False).cuda(device=opt.device)
            utils.progress_bar(i, len(dataloader), 'Eva... ')
            res = cleaner(hazy)
            res = res.data.cpu().numpy()[0]
            res[res > 1] = 1
            res[res < 0] = 0

            # res = res.transpose((1, 2, 0))
            ori = hazy_1.data.cpu().numpy()[0]
            ori *= 255
            ori = ori.astype(np.uint8)
            ori = ori.transpose((1, 2, 0))

            res *= 255
            res = res.astype(np.uint8)
            res = res.transpose((1, 2, 0))
            ims = np.hstack((ori, res))
            # pdb.set_trace()

            # np.vstack()

            Image.fromarray(ims).save(show_dst + get_file_name(im_name[0]) + '.png')

        else:
            print("Unknown dataset name.")


    # print('psnr: ' + str(ave_psnr / float(ct_num)) + '.')
    # print('ssim: ' + str(ave_ssim / float(ct_num)) + '.')


if __name__ == '__main__':
    """
        Dataset
    """
    transform = transforms.ToTensor()

    dataloader = get_data_loader_folder(testroot, 1, train=False, new_size=(512, 512), num_workers=1, crop=False,
                                        return_paths=True)

    model = Model(opt)
    model = model.cuda(device=opt.device)
    model.eval()
    # Make the network

    test(model.cleaner, dataloader, 'SINGLE')
