import pdb
import numpy as np
import torch
import random
import torch.utils.data as data
from PIL import Image
import cv2


class ImageSet(data.Dataset):
    # Init. it.
    def __init__(self, dataroot,
                 imlist_pth,
                 transform=None,
                 resize_to=None,
                 crop_size=None,
                 is_train=False,
                 with_aug=False,
                 max_size=999999):
        self.is_train = is_train
        self.with_aug = with_aug
        self.dataroot = dataroot
        self.transform = transform
        self.resize_to = resize_to
        self.crop_size = crop_size
        self.imlist = self.flist_reader(imlist_pth)
        self.max_size = max_size

    # Process data.
    def __getitem__(self, index):
        im_name = self.imlist[index]
        im_input, label = self.sample_loader(im_name)

        # Resize a sample, or not.
        if not self.resize_to is None:
            im_input = cv2.resize(im_input, self.resize_to)
            label = cv2.resize(label, self.resize_to)

        # Transform: output torch.tensors of [0,1] and (C,H,W).
        # Note: for test on DDN_Data and RESIDE, the output is in [0,1] and (V,C,H,W).
        #       V means the distortation types of a dataset (e.g., V == 14 for DDN_Data)

        if not self.transform is None:
            im_input, label = self.Transformer(im_input, label)

        return im_input, label, im_name

    # Read a image name list.
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist

    # Return a pair of images (input, label).
    def sample_loader(self, im_name):
        return RESIDE_loader(self.dataroot, im_name, self.is_train)

    def Transformer(self, im_input, label):
        if not self.is_train:
            label = self.transform(label)
            im_input = im_input.transpose((3, 2, 0, 1))
            im_input = torch.FloatTensor(im_input)
            im_input /= 255.0
        else:
            if not self.crop_size is None:
                im_input, label = CropSample(im_input, label, self.crop_size)
            if self.with_aug:
                im_input, label = DataAugmentation(im_input, label)

            im_input = self.transform(im_input)
            label = self.transform(label)

        return im_input, label

    def __len__(self):
        return min(len(self.imlist), self.max_size)


def RESIDE_loader(dataroot, im_name, is_train):
    if not is_train:
        Vars = np.arange(1, 11, 1)
        label_pth = dataroot + 'labels/' + im_name
        label = Image.open(label_pth).convert("RGB")
        for var in Vars:
            if var == 1:
                hazy = np.asarray(Image.open(
                    dataroot + 'images/' + im_name.split('.')[0] + '_' + str(var) + '.png'))
                hazy = np.expand_dims(hazy, axis=3)
            else:
                current = np.asarray(Image.open(
                    dataroot + 'images/' + im_name.split('.')[0] + '_' + str(var) + '.png'))
                current = np.expand_dims(current, axis=3)
                hazy = np.concatenate((hazy, current), axis=3)
    else:
        var = random.choice(np.arange(1, 11, 1))
        label_pth = dataroot + 'labels/' + im_name
        hazy_pth = dataroot + 'images/' + im_name.split('.')[0] + '_' + str(var) + '.png'

        label = Image.open(label_pth).convert("RGB")
        hazy = Image.open(hazy_pth).convert("RGB")

    return hazy, label


def CropSample(im_input, label, crop_size):
    if isinstance(label, np.ndarray):
        label = Image.fromarray(label)
    if isinstance(im_input, np.ndarray):
        im_input = Image.fromarray(im_input)

    W, H = label.size
    x_offset = random.randint(0, W - crop_size)
    y_offset = random.randint(0, H - crop_size)
    label = label.crop((x_offset, y_offset,
                        x_offset + crop_size, y_offset + crop_size))
    im_input = im_input.crop((x_offset, y_offset,
                              x_offset + crop_size, y_offset + crop_size))
    return im_input, label


def DataAugmentation(im_input, label):
    if random.random() > 0.5:
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        im_input = im_input.transpose(Image.FLIP_LEFT_RIGHT)
    return im_input, label


def align_to_k(img, k=4):
    a_row = int(img.shape[0] / k) * k
    a_col = int(img.shape[1] / k) * k
    img = img[0:a_row, 0:a_col]
    return img
