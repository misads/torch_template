import argparse
import os
import random
import torch
import os
from torch_template import misc_utils as utils


def parse_args():
    # experiment specifics
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', type=str, default='cache',
                        help='folder name to clear')

    parser.add_argument('--rm', action='store_true', help='debug mode')

    return parser.parse_args()


opt = parse_args()

paths = ['checkpoints', 'logs', 'results']

utils.color_print("Directory '%s' cleared." % opt.tag, 1)
if opt.rm:
    for path in paths:
        p = os.path.join(path, opt.tag)
        if os.path.isdir(p):
            command = 'rm -r ' + p
            print(command)
            os.system(command)
else:
    tmp = os.path.join('trash', str(random.randint(1000000000, 9999999999)))
    utils.try_make_dir(tmp)
    for path in paths:
        p = os.path.join(path, opt.tag)
        if os.path.isdir(p):
            command = 'mv %s %s' % (p, tmp)
            print(command)
            os.system(command)
