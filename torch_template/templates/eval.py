# python 2.7, pytorch 0.3.1

import os, sys
import pdb

from dataloader.tta import OverlapTTA
from loss.cls_loss import f1_loss
from options import opt
from options.options import logger
from utils.torch_utils import write_loss, write_image, tensor2im

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


def evaluate(cleaner, dataloader, epochs, writer, thresh=160, test_mode=False):

    if thresh > 1:
        thresh = thresh / 255
    data_name = ''

    result_dir = os.path.join(opt.result_dir, opt.tag, str(epochs))
    utils.try_make_dir(result_dir)

    ave_f1 = {}
    ct_num = 0
    # print('Start testing ' + tag + '...')
    for i, data in enumerate(dataloader):
        if test_mode:
            img, path = data  # ['label'], data['image']  #
            path = utils.get_file_name(path[0]) + '.png'
        else:
            img, label = data  # ['label'], data['image']  #

        with torch.no_grad():
            img_var = Variable(img, requires_grad=False).cuda(device=opt.device)
            if not test_mode:
                label_var = Variable(label, requires_grad=False).cuda(device=opt.device)
                label_im = np.asarray(tensor2im(label_var))

            tta = OverlapTTA(img, 10, 10, 256, 256, norm_patch=False, flip_aug=False)
            for j, x in enumerate(tta):  # 获取每个patch输入
                utils.progress_bar(i * 100 + j, len(dataloader) * 100, 'Eva...', '')
                generated = cleaner(x)
                torch.cuda.empty_cache()
                tta.collect(generated[0], j)  # 收集inference结果

            ######################################
            #    Denorm & convert to img format
            ######################################
            output = tta.combine()
            output = (output + 1) / 2
            output = output.detach().cpu().numpy()

            if not test_mode:
                for t in range(140, 251, 5):
                    thresh = t / 255
                    output_copy = output.copy()
                    output_copy[output_copy < thresh] = 0
                    output_copy[output_copy >= thresh] = 1

                    f1 = f1_loss(output_copy, label_im)
                    if str(f1) == 'nan':
                        f1 = 0.

                    if t in ave_f1:
                        ave_f1[t] += f1
                    else:
                        ave_f1[t] = f1
                    del output_copy

        ct_num += 1

        if test_mode:
            write_image(writer, 'test/%s' % path, '1_input', tensor2im(img_var), epochs)
            write_image(writer, 'test/%s' % path, '2_output', output, epochs)

            save_path = os.path.join(result_dir, path)

            output *= 255
            res = output.astype(np.uint8)
            res = res.transpose((1, 2, 0))
            Image.fromarray(res).save(save_path)
        elif i < 5:
            write_image(writer, 'val/%d' % i, '1_input', tensor2im(img_var), epochs)
            write_image(writer, 'val/%d' % i, '2_output', output, epochs)
            write_image(writer, 'val/%d' % i, '3_target', label_im, epochs)

        # if data_name == 'RESIDE':
        #     ave_psnr += psnr(res, label_v, data_range=255)
        #     ave_ssim += ski_ssim(res, label_v, data_range=255, multichannel=True)
        #     Image.fromarray(res).save(show_dst + im_name[0].split('.')[0] + '_' + str(vi + 1) + '.png')
        #
        # elif data_name == 'SINGLE':
        #     hazy = data
        #     hazy = Variable(hazy, requires_grad=False).cuda(device=opt.device)
        #     utils.progress_bar(i, len(dataloader), 'Eva... ')
        #     res = cleaner(hazy)
        #     res = res.data.cpu().numpy()[0]
        #     res[res > 1] = 1
        #     res[res < 0] = 0
        #
        #     # res = res.transpose((1, 2, 0))
        #     res *= 255
        #     res = res.astype(np.uint8)
        #     write_image(writer, 'real/%d' % i, 'hazy', hazy.data[0], epochs)
        #     write_image(writer, 'real/%d' % i, 'output', res, epochs)

    # if data_name == 'RESIDE':
    #     write_loss(writer, 'val', 'psnr', ave_psnr / float(ct_num), epochs)
    #     write_loss(writer, 'val', 'ssim', ave_ssim / float(ct_num), epochs)
    #
    #     logger.info('Eva epoch %d ,' % epochs + 'psnr: ' + str(ave_psnr / float(ct_num)) + '.')
    #     logger.info('Eva epoch %d ,' % epochs + 'ssim: ' + str(ave_ssim / float(ct_num)) + '.')

    if not test_mode:
        logger.info('Eva epoch %d ,' % epochs + 'Average F1: ')

        results = sorted(ave_f1.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        for t, v in results:
            write_loss(writer, 'val', 'f1/thresh_%d' % t, v / float(ct_num), epochs)
            logger.info('Thresh: %d F1: %f' % (t, v / float(ct_num)))


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
