###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import pdb
import random

import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import os.path

from dataloader.transforms import __crop, __flip
from utils.torch_utils import create_summary_writer, write_image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def read_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [fn.strip() for fn in fns]
    return fns


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, fns=None, repeat=1):
    """
        :param dir:
        :param fns: To specify file name list
        :param repeat:
        :return:
    """
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    if fns is None:
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    for i in range(repeat):
                        images.append(path)
    else:
        for fname in fns:
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                images.append(path)
                for i in range(repeat):
                    images.append(path)
    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


"""
    ImageFolder
"""


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        try:
            path = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            if self.return_paths:
                return img, path
            else:
                return img
        except:
            print(index)
            pdb.set_trace()

    def __len__(self):
        return len(self.imgs)


# def get_data_loader_list(root, file_list, batch_size, train, new_size=None,
#                            height=256, width=256, num_workers=4, crop=True):
#     transform_list = [transforms.ToTensor(),
#                       transforms.Normalize((0.5, 0.5, 0.5),
#                                            (0.5, 0.5, 0.5))]
#     transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
#     transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
#     transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
#     transform = transforms.Compose(transform_list)
#     dataset = ImageFilelist(root, file_list, transform=transform)
#     loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
#     return loader

def get_data_loader_folder(input_folder, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True, normalization=False, return_paths=False):
    transform_list = [transforms.ToTensor()]

    transform_list = transform_list + [transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))] if normalization else transform_list
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if train and crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform, return_paths=return_paths)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


"""
    PairedImageFolder
"""


def get_params():
    # w, h = size
    # crop_w, crop_h = crop_size
    # x = random.randint(0, np.maximum(0, w - crop_w))
    # y = random.randint(0, np.maximum(0, h - crop_h))
    x = random.random()
    y = random.random()

    flip = random.randint(0, 7)
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(new_size, height, width, crop, flip, normalization, params):
    transform_list = [transforms.ToTensor()]

    transform_list = transform_list + [transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))] if normalization else transform_list

    transform_list = [transforms.Lambda(
        lambda img: __crop(img, params['crop_pos'], (height, width)))] + transform_list if \
        crop else transform_list

    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list

    transform_list = [transforms.Lambda(
        lambda img: __flip(img, params['flip']))] + transform_list if flip else transform_list

    transform = transforms.Compose(transform_list)

    return transform


class PairedImageFolder(data.Dataset):

    def __init__(self, root_A, root_B, new_size, height, width, crop, flip, normalization, return_paths=False,
                 loader=default_loader):
        imgs_A = sorted(make_dataset(root_A))
        imgs_B = sorted(make_dataset(root_B))
        if len(imgs_A) == 0:
            raise (RuntimeError("Found 0 images in: " + root_A + "\n"
                                                                 "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))
        if len(imgs_B) == 0:
            raise (RuntimeError("Found 0 images in: " + root_B + "\n"
                                                                 "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root_A = root_A
        self.root_B = root_B
        self.imgs_A = imgs_A
        self.imgs_B = imgs_B
        self.new_size = new_size
        self.height = height
        self.width = width
        self.crop = crop
        self.flip = flip
        self.normalization = normalization
        assert len(imgs_A) == len(imgs_B), "Images not corresponded between '%s' and '%s'" % (root_A, root_B)
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        # try:
        path_A = self.imgs_A[index]
        path_B = self.imgs_B[index]
        img_A = self.loader(path_A)
        img_B = self.loader(path_B)
        params = get_params()
        transform = get_transform(self.new_size, self.height, self.width, self.crop, self.flip, self.normalization, params)
        img_A = transform(img_A)
        img_B = transform(img_B)
        if self.return_paths:
            return img_A, path_A, img_B, path_B
        else:
            return img_A, img_B
        # except:
        #     print(index)
        #     pdb.set_trace()

    def __len__(self):
        return len(self.imgs_A)


def get_paired_data_loader_folder(input_folder_A, input_folder_B, batch_size, train, new_size=None,
                                  height=256, width=256, num_workers=4, crop=True, normalization=False):

    if not train:
        crop = False
    dataset = PairedImageFolder(input_folder_A, input_folder_B, new_size, height, width, crop, train, normalization)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


if __name__ == '__main__':
    data_root = '/home/xhy/big_money/money_net/datasets/AUG8'
    val_root = '/home/xhy/dataset/neural/complex_256'

    label = 'neural'

    train_A_root = os.path.join(data_root, label, 'train_A')
    train_B_root = os.path.join(data_root, label, 'train_B')
    val_A_root = os.path.join(val_root, 'val_A')
    val_B_root = os.path.join(val_root, 'val_B')

    dataloader = get_paired_data_loader_folder(train_A_root, train_B_root, 1, True, normalization=True)
    print(len(dataloader))
    import numpy as np

    logroot = './logs/'
    # writer = create_summary_writer(logroot)

    for e in range(1):
        for iteration, data in enumerate(dataloader):
            a, b = data
            # write_image(writer, 'train_%d' % i, 'a', a.data[0], e)
            # write_image(writer, 'train_%d' % i, 'b', b.data[0], e)
            # print(a.data[0].shape)

            print(iteration)
