# encoding = utf-8
"""
    U-RISC Challenge

    Team: Microscope
    
    Url:
        https://www.biendata.com/competition/urisc/

    File Structure:
        .
        ├── train.py                :Train and evaluation loop, errors and outputs visualization (Powered by TensorBoard)
        ├── test.py                 :Test
        │
        ├── network
        │     ├── Model.py          :Define models, losses and parameter updating
        │     └── *.py              :Define networks
        ├── options
        │     └── options.py        :Define options
        │
        ├── datasets
        │     ├── simple            :Downloaded datasets
        │     └── complex
        │
        ├── dataloader/             :Define Dataloaders
        ├── backbone                :Commonly used models
        ├── utils
        │     ├── misc_utils.py     :System utils
        │     └── torch_utils.py    :PyTorch utils
        │
        ├── checkpoints/<tag>       :Trained checkpoints
        ├── logs/<tag>              :Logs and TensorBoard event files
        └── results/<tag>           :Test results

    Usage:

    #### Train

        python train.py --tag together_1 --epochs 4000 --weight_bce 20. --weight_dice .5 --weight_focal 0.
            --dataset simple/cleaned --valset simple/big_cheat --gpu_ids 1

    #### Resume or Fine Tune

        python train.py --load checkpoints/network_1 --which-epoch 500  # other args same as ↑

    #### test

        python test.py --tag together_1 --dataset simple/val

    License: MIT

    Last modified 12.28
"""

import os
import pdb
import time

import torch
from torch.autograd import Variable

from data.data_loader import CreateDataLoader
from dataloader import get_paired_data_loader_folder, get_data_loader_folder
from eval import evaluate
from network.Model import Model
from options import opt, logger
from utils.torch_utils import create_summary_writer, write_image, write_meters_loss, LR_Scheduler
import utils.misc_utils as utils
import numpy as np
dataset = os.path.join(opt.data_root, opt.dataset)
if not os.path.isdir(dataset):
    raise FileNotFoundError("Dataset '%s' not found" % dataset)

######################
#  Training dataset
######################
utils.color_print("Training dataset is set to '%s'" % dataset, 3)

train_A_path = os.path.join(dataset, 'train_A')
train_B_path = os.path.join(dataset, 'train_B')

train_dataloader = get_paired_data_loader_folder(train_A_path, train_B_path, batch_size=opt.batch_size,
                                                 train=True, height=opt.crop, width=opt.crop, num_workers=1,
                                                 crop=True, flip=True, normalization=True)

test_path = os.path.join(dataset, 'test_A')
has_test = False
if os.path.isdir(test_path):
    has_test = True
    utils.color_print("Test dataset is set to '%s'" % test_path, 3)
    test_dataloader = get_data_loader_folder(test_path, batch_size=1, train=False, num_workers=1,
                                             crop=False, normalization=True, return_paths=True)

######################
#  Validation dataset
######################
has_val = opt.valset is not None

if has_val:
    val_set = os.path.join(opt.data_root, opt.valset)
    if not os.path.isdir(val_set):
        raise FileNotFoundError("Validation dataset '%s' not found" % val_set)

    val_A_path = os.path.join(val_set, 'train_A')
    val_B_path = os.path.join(val_set, 'train_B')

    utils.color_print("Validation dataset is set to '%s'" % val_set, 2)

    val_dataloader = get_paired_data_loader_folder(val_A_path, val_B_path, batch_size=1,
                                                   train=False, height=opt.crop, width=opt.crop, num_workers=1,
                                                   crop=False, flip=False, normalization=True)

    # val_dataloader = get_data_loader_folder(test_path, batch_size=1, train=False, num_workers=1,
    #                                          crop=False, normalization=True, return_paths=True)


if opt.debug:
    opt.save_freq = 1
    opt.eval_freq = 1
    opt.log_freq = 1

######################
#       Paths
######################
save_root = os.path.join(opt.checkpoint_dir, opt.tag)
log_root = os.path.join(opt.log_dir, opt.tag)

utils.try_make_dir(save_root)
utils.try_make_dir(log_root)



######################
#     Init model
######################
model = Model(opt)

# if len(opt.gpu_ids):
#     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model = model.cuda(device=opt.device)

start_epoch = opt.which_epoch if opt.which_epoch else 0
model.train()

# Start training
print('Start training...')
start_step = start_epoch * len(train_dataloader)
global_step = start_step
total_steps = opt.epochs * len(train_dataloader)
start = time.time()

#####################
#   cosine 学习率
#####################
scheduler = LR_Scheduler('cos', opt.lr, opt.epochs, len(train_dataloader), warmup_epochs=0, logger=logger)
######################
#    Summary_writer
######################
writer = create_summary_writer(log_root)

######################
#     Train loop
######################
for epoch in range(start_epoch, opt.epochs):
    for iteration, data in enumerate(train_dataloader):
        ####################
        #     Update lr
        ####################
        scheduler(model.g_optimizer, iteration, epoch + 1)

        global_step += 1
        rate = (global_step - start_step) / (time.time() - start)
        remaining = (total_steps - global_step) / rate

        img, label = data  # ['label'], data['image']  #
        img_var = Variable(img, requires_grad=False).cuda(device=opt.device)
        label_var = Variable(label, requires_grad=False).cuda(device=opt.device)

        # Cleaning noisy images
        cleaned = model.cleaner(img_var)

        ##############################
        #       Update parameters
        ##############################
        model.update_G(cleaned, label_var)

        ###############################
        #     Tensorboard Summary
        ###############################
        if epoch % opt.log_freq == opt.log_freq - 1 and iteration < 5:
            img = (img.detach().cpu() + 1) / 2
            # cleaned = cleaned.mean(dim=1, keepdim=True)
            cleaned = (cleaned.detach().cpu() + 1) / 2
            label_var = (label_var.detach().cpu() + 1) / 2

            write_image(writer, 'train/%d' % iteration, 'input', img.data[0], epoch)
            cleaned[cleaned > 1] = 1
            cleaned[cleaned < 0] = 0
            write_image(writer, 'train/%d' % iteration, 'output', cleaned.data[0], epoch)
            write_image(writer, 'train/%d' % iteration, 'target', label_var.data[0], epoch)

        pre_msg = 'Epoch:%d' % epoch

        msg = '(loss) %s ETA: %s' % (str(model.avg_meters), utils.format_time(remaining))
        utils.progress_bar(iteration, len(train_dataloader), pre_msg, msg)
        # print(pre_msg, msg)

    write_meters_loss(writer, 'train', model.avg_meters, epoch)
    logger.info('Train epoch %d, (loss) ' % epoch + str(model.avg_meters))

    if epoch % opt.save_freq == opt.save_freq - 1 or epoch == opt.epochs - 1:  # 每隔10次save checkpoint
        model.save(epoch)

    ####################
    #     Validation
    ####################
    if epoch % opt.eval_freq == (opt.eval_freq - 1):

        model.eval()
        if has_val:
            # evaluate(model.cleaner, val_dataloader, epoch + 1, writer)
            pass
        if has_test:
            evaluate(model.cleaner, test_dataloader, epoch + 1, writer, test_mode=True)
            utils.color_print("Test results have been saved.", 3)
        model.train()

