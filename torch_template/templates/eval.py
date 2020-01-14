# python 2.7, pytorch 0.3.1

import os, sys
import pdb

from torch_template.dataloader.tta import OverlapTTA
# from torch_template.loss.cls_loss import f1_loss
from options import opt
from options.options import logger
from torch_template.utils.torch_utils import write_loss, write_image, tensor2im

sys.path.insert(1, '../')
import torch
import torchvision
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image

from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ski_ssim
import misc_utils as utils


def evaluate(cleaner, dataloader, epochs, writer, test_mode=False, norm=False):

    result_dir = os.path.join(opt.result_dir, opt.tag, str(epochs))
    utils.try_make_dir(result_dir)

    ave_psnr = 0.
    ave_ssim = 0.
    ct_num = 0
    # print('Start testing ' + tag + '...')
    for i, data in enumerate(dataloader):
        if test_mode:
            img, path = data  # ['label'], data['image']  #
            path = utils.get_file_name(path[0]) + '.png'
        else:
            img, label, _ = data  # ['label'], data['image']  #

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
            output[output > 1] = 1
            output[output < 0] = 0
            output = output.detach().cpu().numpy()

        if test_mode:
            write_image(writer, 'test/%s' % path, '1_input', tensor2im(img_var, norm=norm), epochs)
            write_image(writer, 'test/%s' % path, '2_output', output, epochs)

            save_path = os.path.join(result_dir, path)

            output *= 255
            res = output.astype(np.uint8)
            res = res.transpose((1, 2, 0))
            Image.fromarray(res).save(save_path)

        else:
            if i < 5:
                write_image(writer, 'val/%d' % i, '1_input', tensor2im(img_var, norm=norm), epochs)
                write_image(writer, 'val/%d' % i, '2_output', output, epochs)
                write_image(writer, 'val/%d' % i, '3_target', label_im, epochs)

            res = output.transpose((1, 2, 0))
            label_im = label_im.transpose((1, 2, 0))
            ave_psnr += psnr(res, label_im)
            ave_ssim += ski_ssim(res, label_im, multichannel=True)
            ct_num += 1


    if not test_mode:
        write_loss(writer, 'val', 'psnr', ave_psnr / float(ct_num), epochs)
        write_loss(writer, 'val', 'ssim', ave_ssim / float(ct_num), epochs)

        logger.info('Eva epoch %d ,' % epochs + 'psnr: ' + str(ave_psnr / float(ct_num)) + '.')
        logger.info('Eva epoch %d ,' % epochs + 'ssim: ' + str(ave_ssim / float(ct_num)) + '.')


    # Set transformer, convertor, and data_loader
    from torch_template.dataloader.image_folder import get_data_loader_folder
    from options import opt
    from network import models
    import misc_utils as utils
    from torch_template.utils.torch_utils import create_summary_writer

    log_root = os.path.join(opt.result_dir, opt.tag)
    utils.try_make_dir(log_root)

    HSTS_PATH = './datasets/RESIDE/HSTS/synthetic_png'
    hsts_dataset = HSTSDataset(HSTS_PATH)
    hsts_dataloader = DataLoader(hsts_dataset, batch_size=1, shuffle=False, num_workers=1)

    realroot = "./datasets/" + data_name + "/REAL/"
    real_dataloader = get_data_loader_folder(realroot, 1, train=False, num_workers=1, crop=False)

    Model = models[opt.model]
    model = Model(opt)
    model = model.cuda(device=opt.device)

    writer = create_summary_writer(log_root)

    evaluate(model.cleaner, hsts_dataloader, opt.which_epoch + 1, writer)
    evaluate(model.cleaner, real_dataloader, opt.which_epoch + 1, writer, 'SINGLE')


if __name__ == '__main__':
    # Set transformer, convertor, and data_loader
    from torch_template.dataloader.image_folder import get_data_loader_folder
    from options import opt
    from network import get_model
    import misc_utils as utils
    from torch_template.utils.torch_utils import create_summary_writer

    log_root = os.path.join(opt.result_dir, opt.tag)
    utils.try_make_dir(log_root)

    realroot = "./datasets/" + data_name + "/REAL/"
    real_dataloader = get_data_loader_folder(realroot, 1, train=False, num_workers=1, crop=False)

    Model = get_model(opt.model)
    model = Model(opt)
    model = model.cuda(device=opt.device)

    writer = create_summary_writer(log_root)

    evaluate(model.cleaner, real_dataloader, opt.which_epoch + 1, writer)

