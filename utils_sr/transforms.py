import torch
import random
import numpy as np

from torchvision import transforms as T
from torchvision.transforms import functional as F

"""
    相关的参数说明：
    Compose             ：增强组合
    ToTensor            ：类型转换
    Normalize           ：标准化
    RandomCrop          ：随机裁剪
    CenterCrop          ：中心裁剪
    RandomResize        ：随机放缩
    RandomHorizontalFlip：随机翻转
"""


def pad_if_smaller(img, size, fill=0):
    # 图像高宽小于给定值,采用 fill 进行填充
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        pad_h = size - oh if oh < size else 0
        pad_w = size - ow if ow < size else 0
        img = F.pad(img, [0, 0, pad_w, pad_h], fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for trans in self.transforms:
            image, target = trans(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, [size])
        # 在 torchvision(0.9.0) 以后才有 InterpolationMode.NEAREST
        # 如果是之前的版本需要使用 PIL.Image.NEAREST
        target = F.resize(target, [size], interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
