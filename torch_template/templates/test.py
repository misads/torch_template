# encoding=utf-8
"""
  python3 test.py --tag complex_256 --dataset complex --load checkpoints/complex --which-epoch 499

"""

import os, sys
import pdb

from dataloader.image_folder import get_data_loader_folder
from dataloader.tta import OverlapTTA
from network.Model import Model
from options import opt
from misc_utils import get_file_name

sys.path.insert(1, '../')
import torch
import torchvision
import numpy as np
from PIL import Image

import misc_utils as utils

# ######################
# #       Dataset
# ######################
dataset = os.path.join(opt.data_root, opt.dataset)
if not os.path.isdir(dataset):
    raise FileNotFoundError("Dataset '%s' not found" % dataset)

utils.color_print("Dataset is set to '%s'" % dataset, 3)

test_path = os.path.join(dataset, 'test_A')

# train_dataloader = get_paired_data_loader_folder(train_A_path, train_B_path, batch_size=opt.batch_size,
#                                                  train=True, height=opt.crop, width=opt.crop, num_workers=1,
#                                                  crop=True, flip=True, normalization=True)
test_dataloader = get_data_loader_folder(test_path, batch_size=1, train=False, num_workers=1,
                                         crop=False, normalization=True, return_paths=True)


if not opt.load:
    raise Exception('Checkpoint must be specified at test phase, try --load <checkpoint_dir>')


result_dir = os.path.join(opt.result_dir, opt.tag)
utils.try_make_dir(result_dir)

model = Model(opt)

model = model.cuda(device=opt.device)
model.eval()


for i, data in enumerate(test_dataloader):
    print('Testing image %d' % i)
    img, paths = data
    filename = get_file_name(paths[0])
    tta = OverlapTTA(img, 100, 100, 256, 256, norm_patch=False, flip_aug=False)
    for j, x in enumerate(tta):  # 获取每个patch输入
        utils.progress_bar(j, len(tta))
        generated = model.cleaner(x)
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

    output *= 255
    res = output.astype(np.uint8)
    res = res.transpose((1, 2, 0))

    ######################################
    #            Save the result
    ######################################
    save_path = os.path.join(result_dir, filename + '.png')

    Image.fromarray(res).save(save_path)

