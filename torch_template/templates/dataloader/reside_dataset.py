import os.path
from os.path import join
from dataloader.image_folder import make_dataset
from dataloader.transforms import Sobel, to_norm_tensor, to_tensor, ReflectionSythesis_1, ReflectionSythesis_2, \
    paired_data_transforms
from PIL import Image
import random
import torch
import math
import pdb
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# import util.util as util
#import data.torchdata as torchdata
import torch.utils.data as data


# BaseDataset = torchdata.Dataset
BaseDataset = data.Dataset


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print('Reset Dataset...')
            self.dataset.reset()


class ITSDataset(BaseDataset):
    """
        Indoor training dataset
        Example:
            train_dataset = datasets.ITSDataset(
            datadir_h, datadir_c, datadir_t, size=max_dataset_size, enable_transforms=True)

            train_dataset[index]:
            return: {
                'hazy': hazy image,
                'clear': clear image (gt),
                't': transmission map image,
                'filename': hazy file name,
                'hazy_path': hazy file path,
                'clear_path': clear file path,
                't_path': t file path ,
            }
    """
    def __init__(self, datadir_h, datadir_c, datadir_t, fns=None, size=None, enable_transforms=True):
        super(ITSDataset, self).__init__()
        self.size = size
        self.datadir_h = datadir_h
        self.datadir_c = datadir_c
        self.datadir_t = datadir_t
        self.enable_transforms = enable_transforms
        #
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths_h = sorted(make_dataset(datadir_h, fns), key=lambda x: x.split('_')[0] + x.split('_')[1])
        self.paths_c = sorted(make_dataset(datadir_c, fns, repeat=10), key=lambda x: x.split('.')[0])
        self.paths_t = sorted(make_dataset(datadir_t, fns), key=lambda x: x.split('_')[0] + x.split('_')[1])
        for i in range(len(self.paths_c)):
            self.paths_c[i] = self.paths_t[i].replace('trans', 'clear').split('_')[0] + '.png'
        if size is not None:
            self.paths_h = self.paths_h[:size]
            self.paths_c = self.paths_c[:size]
            self.paths_t = self.paths_t[:size]

        self.reset(shuffle=False)

    def reset(self, shuffle=True):
        if shuffle:
            # c = list(zip(a, b))
            # random.shuffle(c)
            # a[:], b[:] = zip(*c)
            paths = list(zip(self.paths_h, self.paths_c, self.paths_t))
            random.shuffle(paths)
            self.paths_h[:], self.paths_c[:], self.paths_t[:] = zip(*paths)
        # num_paths = len(self.paths) // 2
        self.H_paths = self.paths_h
        self.C_paths = self.paths_c
        self.T_paths = self.paths_t

    def __getitem__(self, index):
        index_H = index % len(self.H_paths)
        index_C = index % len(self.C_paths)
        index_T = index % len(self.T_paths)

        H_path = self.H_paths[index_H]
        C_path = self.C_paths[index_C]
        T_path = self.T_paths[index_T]

        h_img = Image.open(H_path).resize((256, 256)).convert('RGB')
        c_img = Image.open(C_path).resize((256, 256)).convert('RGB')
        t_img = Image.open(T_path).resize((256, 256)).convert('RGB')
        H = to_tensor(h_img)
        C = to_tensor(c_img)
        T = to_tensor(t_img)

        fn = os.path.basename(H_path)
        return {'hazy': H, 'clear': C, 't': T, 'hazy_path': H_path, 'clear_path': C_path,
                't_path': T_path}

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.H_paths), len(self.C_paths)), self.size)
        else:
            return max(len(self.H_paths), len(self.C_paths))


class SOTSTestDataset(BaseDataset):
    def __init__(self, datadir_h, datadir_c, fns=None, size=None, enable_transforms=False, unaligned_transforms=False,
                 round_factor=1, flag=None):
        super(SOTSTestDataset, self).__init__()
        self.size = size
        self.datadir_h = datadir_h
        self.datadir_c = datadir_c
        self.fns = fns or os.listdir(datadir_h)
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths_h = sorted(make_dataset(datadir_h, fns), key=sortkey)
        self.paths_c = sorted(make_dataset(datadir_c, fns, repeat=10), key=sortkey)

        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]

        h_img = Image.open(self.paths_h[index]).convert('RGB')
        c_img = Image.open(self.paths_c[index]).convert('RGB')
        # c_img = Image.open(self.paths_c[index]).crop((10, 10, 630, 470)).convert('RGB')  # 裁剪

        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(h_img, c_img, self.unaligned_transforms)

        H = to_tensor(h_img)
        C = to_tensor(c_img)

        dic = {'hazy': H, 'clear': C, 'hazy_path': self.paths_h[index], 'clear_path': self.paths_c[index]}
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)
