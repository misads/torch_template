import os
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataloader import dual_residual_dataset
from dataloader.image_folder import get_data_loader_folder
from eval import evaluate
from network.Model_Dual import Model
from options import opt, logger
from utils.torch_utils import create_summary_writer, write_image, write_meters_loss, LR_Scheduler
import utils.misc_utils as utils
import torch

data_name = 'RESIDE'
data_root = './datasets/' + data_name + '/ITS/'
imlist_pth = './datasets/' + data_name + '/indoor_train_list.txt'
valroot = "./datasets/" + data_name + "/SOTS/nyuhaze500/"
val_list_pth = './datasets/' + data_name + '/sots_test_list.txt'
realroot = "./datasets/" + data_name + "/REAL/"
real_list_pth = './datasets/' + data_name + '/real.txt'

# dstroot for saving models.
# logroot for writting some log(s), if is needed.
save_root = os.path.join(opt.checkpoint_dir, opt.tag)
log_root = os.path.join(opt.log_dir, opt.tag)

utils.try_make_dir(save_root)
utils.try_make_dir(log_root)

if opt.debug:
    opt.save_freq = 1
    opt.eval_freq = 1
    opt.log_freq = 1

# Transform
transform = transforms.ToTensor()
# Dataloader
max_size = 9999999
if opt.debug:
    max_size = opt.batch_size * 10

train_dataset = dual_residual_dataset.ImageSet(data_root, imlist_pth,
                                               transform=transform, is_train=True,
                                               with_aug=False, crop_size=opt.crop, max_size=max_size)
dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=5)
"""
    Val dataset
"""
val_dataset = dual_residual_dataset.ImageSet(valroot, val_list_pth,
                                             transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
"""
    Real val dataset
"""
real_dataloader = get_data_loader_folder(realroot, 1, train=False, num_workers=1, crop=False)


model = Model(opt)

# if len(opt.gpu_ids):
#     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model = model.cuda(device=opt.device)

# optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
# optimizer_G = model.g_optimizer

start_epoch = opt.which_epoch if opt.which_epoch else 0
model.train()

# Start training
print('Start training...')
start_step = start_epoch * len(dataloader)
global_step = start_step
total_steps = opt.epochs * len(dataloader)
start = time.time()

writer = create_summary_writer(log_root)
#scheduler = LR_Scheduler('cos', opt.lr, opt.epochs, len(dataloader), warmup_epochs=10)

for epoch in range(start_epoch, opt.epochs):
    for iteration, data in enumerate(dataloader):
        global_step += 1
        rate = (global_step - start_step) / (time.time() - start)
        remaining = (total_steps - global_step) / rate

        img, label, _ = data
        img_var = Variable(img, requires_grad=False).cuda(device=opt.device)
        label_var = Variable(label, requires_grad=False).cuda(device=opt.device)

        # Cleaning noisy images
        cleaned, A, t = model.cleaner(img_var)
        model.update_G(cleaned, label_var)

        Jt = torch.clamp(cleaned * t, min=.01, max=.99)
        airlight = torch.clamp(A * (1-t), min=.01, max=.99)

        if epoch % opt.log_freq == opt.log_freq - 1 and iteration < 5:
            write_image(writer, 'train/%d' % iteration, '0_input', img.data[0], epoch)
            write_image(writer, 'train/%d' % iteration, '1_A', A.data[0], epoch)
            write_image(writer, 'train/%d' % iteration, '2_t', t.data[0], epoch)
            write_image(writer, 'train/%d' % iteration, '3_Jt', Jt.data[0], epoch)
            write_image(writer, 'train/%d' % iteration, '4_airlit', airlight.data[0], epoch)
            cleaned[cleaned > 1] = 1
            cleaned[cleaned < 0] = 0
            write_image(writer, 'train/%d' % iteration, '5_output', cleaned.data[0], epoch)
            write_image(writer, 'train/%d' % iteration, '6_target', label_var.data[0], epoch)

        # update


        pre_msg = 'Epoch:%d' % epoch

        msg = '(loss) %s ETA: %s' % (str(model.avg_meters), utils.format_time(remaining))
        utils.progress_bar(iteration, len(dataloader), pre_msg, msg)
        # print(pre_msg, msg)
        #scheduler(model.g_optimizer, iteration, epoch)
        # print('Epoch(' + str(epoch + 1) + '), iteration(' + str(iteration + 1) + '): ' +'%.4f, %.4f' % (-ssim_loss.item(),
        #                                                                                                l1_loss.item()))

    # write_loss(writer, 'train', 'F1', 0.78, epoch)
    write_meters_loss(writer, 'train', model.avg_meters, epoch)
    logger.info('Train epoch %d, (loss) ' % epoch + str(model.avg_meters))

    if epoch % opt.save_freq == opt.save_freq - 1 or epoch == opt.epochs-1:  # 每隔10次save checkpoint
        model.save(epoch)

    if epoch % opt.eval_freq == (opt.eval_freq - 1):
        model.eval()
        evaluate(model.cleaner, val_dataloader, epoch + 1, writer)
        # evaluate(model.cleaner, real_dataloader, epoch + 1, writer, 'SINGLE')
        model.train()


    # if epoch in [700, 1400]:
    #     for param_group in model.g_optimizer.param_groups:
    #         param_group['lr'] *= 0.1

