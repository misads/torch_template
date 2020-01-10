import argparse
import os

import torch

import misc_utils as utils

"""
    Arg parse
    opt = parse_args()
"""


def parse_args():
    # experiment specifics
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', type=str, default='cache',
                        help='folder name to save the outputs')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--log_dir', type=str, default='./logs', help='logs are saved here')
    parser.add_argument('--result_dir', type=str, default='./results', help='results are saved here')

    parser.add_argument('--model', type=str, default='Nested', help='which model to use')
    parser.add_argument('--norm', type=str, default='instance',
                        help='[instance] normalization or [batch] normalization')

    # input/output sizes
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--resize', type=int, default=None, help='scale images to this size')
    parser.add_argument('--crop', type=int, default=256, help='then crop to this size')

    # for datasets
    parser.add_argument('--data_root', type=str, default='./datasets/')
    parser.add_argument('--dataset', type=str, default='train')
    parser.add_argument('--val_set', type=str, default=None)
    parser.add_argument('--test_set', type=str, default=None)


    # loss weight
    parser.add_argument('--weight_bce', type=float, default=20)
    parser.add_argument('--weight_dice', type=float, default=0.5)

    # training options
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--load', type=str, default=None, help='load checkpoint')
    parser.add_argument('--which-epoch', type=int, default=0, help='which epoch to resume')
    parser.add_argument('--epochs', type=int, default=500, help='epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')

    parser.add_argument('--save_freq', type=int, default=50, help='freq to save models')
    parser.add_argument('--eval_freq', '--val_freq', type=int, default=50, help='freq to eval models')
    parser.add_argument('--log_freq', type=int, default=1, help='freq to vis in tensorboard')

    return parser.parse_args()


opt = parse_args()

opt.device = 'cuda:' + opt.gpu_ids if torch.cuda.is_available() and opt.gpu_ids != '-1' else 'cpu'

log_dir = os.path.join(opt.log_dir, opt.tag)
utils.try_make_dir(log_dir)
logger = utils.get_logger(f=os.path.join(log_dir, 'log.txt'), level='info')

logger.info('==================Options==================')
for k, v in opt._get_kwargs():
    logger.info(str(k) + '=' + str(v))
logger.info('===========================================')
# utils.print_args(opt)
