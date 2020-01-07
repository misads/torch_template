# python 2.7, pytorch 0.3.1

import os, sys
import pdb

from options import opt
from options.options import logger
from utils.torch_utils import write_loss, write_image

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
tag = 'DuRN-US'
# Choose a dataset.
data_name = 'RESIDE'  # 'DCPDNData' or 'RESIDE'
# -----------------------

if data_name == 'RESIDE':
    testroot = "../data/" + data_name + "/sots_indoor_test/"
    test_list_pth = "../lists/RESIDE_indoor/sots_test_list.txt"
elif data_name == 'DCPDNData':
    testroot = "../data/" + data_name + "/TestA/"
    test_list_pth = '../lists/' + data_name + '/testA_list.txt'
else:
    print('Unknown dataset name.')

Pretrained = '../trainedmodels/' + data_name + '/' + tag + '_model.pt'


def evaluate(cleaner, dataloader, epochs, writer, data_name='RESIDE'):
    show_dst = '../cleaned_images/' + data_name + '/' + tag + '/' + str(epochs) + '/'
    subprocess.check_output(['mkdir', '-p', show_dst])

    ave_psnr = 0.0
    ave_ssim = 0.0
    ct_num = 0
    # print('Start testing ' + tag + '...')
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
                    utils.progress_bar(i * v + vi, len(dataloader) * v, 'Eva... ')
                    ct_num += 1
                    hazy_vi = hazy[bi, vi]
                    hazy_vi = hazy_vi.unsqueeze(dim=0)
                    hazy_vi = Variable(hazy_vi, requires_grad=False).cuda(device=opt.device)
                    res, A ,t = cleaner(hazy_vi)
                    res = res.data.cpu().numpy()[0]
                    res[res > 1] = 1
                    res[res < 0] = 0
                    res *= 255
                    res = res.astype(np.uint8)
                    res = res.transpose((1, 2, 0))
                    if i == 1:
                        write_image(writer, 'val/%d' % vi, 'input', hazy_vi[0], epochs)
                        write_image(writer, 'val/%d' % vi, 'output', res, epochs, 'HWC')
                        write_image(writer, 'val/%d' % vi, 'target', label_v, epochs, 'HWC')

                    ave_psnr += psnr(res, label_v, data_range=255)
                    ave_ssim += ski_ssim(res, label_v, data_range=255, multichannel=True)
                    Image.fromarray(res).save(show_dst + im_name[0].split('.')[0] + '_' + str(vi + 1) + '.png')

        elif data_name == 'DCPDNData':
            hazy, label, im_name = data
            ct_num += 1
            label = label.numpy()[0]
            label = label.transpose((1, 2, 0))
            hazy = Variable(hazy, requires_grad=False).cuda(device=opt.device)
            res = cleaner(hazy)
            res = res.data.cpu().numpy()[0]
            res[res > 1] = 1
            res[res < 0] = 0
            res = res.transpose((1, 2, 0))
            ave_psnr += psnr(res, label, data_range=1)
            ave_ssim += ski_ssim(res, label, data_range=1, multichannel=True)

            res *= 255
            res = res.astype(np.uint8)
            Image.fromarray(res).save(show_dst + im_name[0].split('.')[0] + '.png')

        elif data_name == 'SINGLE':
            hazy = data
            hazy = Variable(hazy, requires_grad=False).cuda(device=opt.device)
            utils.progress_bar(i, len(dataloader), 'Eva... ')
            res, A, t = cleaner(hazy)
            res = res.data.cpu().numpy()[0]
            res[res > 1] = 1
            res[res < 0] = 0

            # res = res.transpose((1, 2, 0))
            res *= 255
            res = res.astype(np.uint8)
            write_image(writer, 'real/%d' % i, 'hazy', hazy.data[0], epochs)
            write_image(writer, 'real/%d' % i, 'output', res, epochs)

        else:
            print("Unknown dataset name.")

    if data_name == 'RESIDE':
        write_loss(writer, 'val', 'psnr', ave_psnr / float(ct_num), epochs)
        write_loss(writer, 'val', 'ssim', ave_ssim / float(ct_num), epochs)

        logger.info('Eva epoch %d ,' % epochs + 'psnr: ' + str(ave_psnr / float(ct_num)) + '.')
        logger.info('Eva epoch %d ,' % epochs + 'ssim: ' + str(ave_ssim / float(ct_num)) + '.')


if __name__ == '__main__':
    # Set transformer, convertor, and data_loader
    transform = transforms.ToTensor()
    convertor = data_convertors.ConvertImageSet(testroot, test_list_pth, data_name,
                                                transform=transform)
    dataloader = DataLoader(convertor, batch_size=1, shuffle=False, num_workers=1)

    # Make the network
    cleaner = cleaner().cuda(device=opt.device)
    cleaner.load_state_dict(torch.load(Pretrained))
    cleaner.eval()
    eval(cleaner, dataloader)
