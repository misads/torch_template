import pdb

import torchvision.transforms.functional as F
import os
from PIL import Image
import torch.utils.data.dataset as dataset
from torchvision import transforms
import random

from torch_template import torch_utils

def paired_cut(img_1: Image.Image, img_2: Image.Image, crop_size):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    r = random.randint(-1, 6)
    if r >= 0:
        img_1 = img_1.transpose(r)
        img_2 = img_2.transpose(r)

    i, j, h, w = get_params(img_1, crop_size)
    img_1 = F.crop(img_1, i, j, h, w)
    img_2 = F.crop(img_2, i, j, h, w)

    return img_1, img_2


class ImageDataset(dataset.Dataset):
    """ImageDataset for training.

    Args:
        datadir(str): dataset root path, default input and label dirs are 'input' and 'gt'
        crop(None, int or tuple): crop size
        aug(bool): data argument (×8)
        norm(bool): normalization

    Example:
        train_dataset = ImageDataset('train', crop=256)
        for i, data in enumerate(train_dataset):
            input, label, file_name = data

    """

    def __init__(self, datadir, crop=None, aug=True, norm=False):
        self.input_path = os.path.join(datadir, 'input')
        self.label_path = os.path.join(datadir, 'gt')
        self.im_names = sorted(os.listdir(self.input_path))
        self.label_names = sorted(os.listdir(self.label_path))

        self.trans_dict = {0: Image.FLIP_LEFT_RIGHT, 1: Image.FLIP_TOP_BOTTOM, 2: Image.ROTATE_90, 3: Image.ROTATE_180,
                           4: Image.ROTATE_270, 5: Image.TRANSPOSE, 6: Image.TRANSVERSE}

        if type(crop) == int:
            crop = (crop, crop)

        self.crop = crop
        self.aug = aug
        self.norm = norm

    def __getitem__(self, index):
        """Get indexs by index

        Args:
            index(int): index

        Returns:
            (tuple): input, label, file_name

        """
        assert self.im_names[index] == self.label_names[index], 'input and label filename not matching.'
        input = Image.open(os.path.join(self.input_path, self.im_names[index])).convert("RGB")
        label = Image.open(os.path.join(self.label_path, self.label_names[index])).convert("RGB")

        if self.crop is not None:
            input, label = paired_cut(input, label, self.crop)
        else:
            input, label = input, label

        r = random.randint(0, 7)
        if self.aug and r != 7:
            input = input.transpose(self.trans_dict[r])
            label = label.transpose(self.trans_dict[r])

        if self.norm:
            input = F.normalize(F.to_tensor(input), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            input = F.to_tensor(input)

        label = F.to_tensor(label)
        return input, label, self.im_names[index]

    def __len__(self):
        return len(os.listdir(self.input_path))


class ImageTestDataset(dataset.Dataset):
    """ImageDataset for test.

    Args:
        datadir(str): dataset path'
        norm(bool): normalization

    Example:
        test_dataset = ImageDataset('test', crop=256)
        for i, data in enumerate(test_dataset):
            input, file_name = data

    """
    def __init__(self, datadir, norm=False):
        self.input_path = datadir
        self.norm = norm
        self.im_names = sorted(os.listdir(self.input_path))

    def __getitem__(self, index):
        # im_name = sorted(os.listdir(self.input_path))[index]
        input = Image.open(os.path.join(self.input_path, self.im_names[index])).convert("RGB")
        if self.norm:
            input = F.normalize(F.to_tensor(input), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            input = F.to_tensor(input)
        return input, self.im_names[index]

    def __len__(self):
        return len(os.listdir(self.input_path))


if __name__ == '__main__':
    writer = torch_utils.create_summary_writer('logs')

    train_dataset = ImageDataset('../datasets/simple/cleaned', crop=None, aug=False, norm=False)

    test_dataset = ImageTestDataset('../datasets/simple/test_A', norm=False)

    # for i, data in enumerate(train_dataset):
    #     input, label, file_name = data
    #     torch_utils.write_image(writer, 'train', '0_input', input, i)
    #     torch_utils.write_image(writer, 'train', '2_label', label, i)
    #     print(i, file_name)

    # for i, data in enumerate(test_dataset):
    #     input, file_name = data
    #     torch_utils.write_image(writer, 'train', file_name, input, i)



