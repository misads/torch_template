from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION

try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import scipy.stats as st
import cv2
import numbers
import types
import collections
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from scipy.signal import convolve2d

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    th, tw = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img: Image.Image, flip):
    """
        :param img:
        :param flip: 0-7, 0 is not flip
        :return:
    """
    flip = flip - 1
    if flip > 0:  # 0-6
        return img.transpose(flip)

    return img
