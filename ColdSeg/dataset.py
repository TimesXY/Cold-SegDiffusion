import os
import torch
import random
import numpy as np
import pandas as pd
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils.data import Dataset

# 环境设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ISICDataset(Dataset):
    def __init__(self, data_path, dataset, is_folder, transform=None, training=True, flip_p=0.0):

        # Read txt file
        fh_txt = open(data_path + "\\" + dataset, 'r')
        images_labels = []
        for line in fh_txt:
            line = line.rstrip()  # Default deletion is blank characters ('\n', '\r', '\t', ' ')
            images_labels.append(line)

        # Assignment and setting of hyperparameters
        self.flip_p = flip_p
        self.training = training
        self.transform = transform

        # Get data path and training image path
        self.data_path = data_path + "\\" + is_folder
        self.images_labels = images_labels

    def __len__(self):
        return len(self.images_labels)

    def __getitem__(self, index):

        # Get the name and absolute path of the image to be read
        name = self.images_labels[index] + '.jpg'
        img_path = os.path.join(self.data_path, name)

        # Get the segmentation mask and absolute path of the image to be read
        mask_name = self.images_labels[index] + '_Segmentation.png'
        msk_path = os.path.join(self.data_path, mask_name)

        # Format conversion of images and masks
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('RGB')

        # Data enhancement process
        if self.transform:
            # Save the randomized state so that the same transforms will be applied to masks
            # and images when finer transforms are used
            state = torch.get_rng_state()
            torch.set_rng_state(state)

            # Data augmentation
            img = self.transform(img)
            mask = self.transform(mask)

            # Random rotation
            if random.random() < self.flip_p:
                img = F.vflip(img)
                mask = F.vflip(mask)

        if self.training:
            return img, mask

        return img, mask, mask_name


class GenericNpyDataset(torch.utils.data.Dataset):
    def __init__(self, directory: str, transform, test_flag: bool = True):
        """ Generic dataset for loading npy files, npy stores 3D arrays, channel 0 is the image, channel 1 is the label.
        """

        super().__init__()
        self.transform = transform
        self.test_flag = test_flag
        self.directory = os.path.expanduser(directory)
        self.filenames = [x for x in os.listdir(self.directory) if x.endswith('.npy')]

    def __getitem__(self, x: int):
        # Access to the name of the document
        file_name = self.filenames[x]

        # Read file
        npy_img = np.load(os.path.join(self.directory, file_name))

        # 获取图像, 并进行维度交换 通道在前
        img = npy_img[:, :, :1]
        img = torch.from_numpy(img).permute(2, 0, 1)

        # Get the mask and binarize it
        mask = npy_img[:, :, 1:]
        mask = np.where(mask > 0, 1, 0)

        # Format conversion of images and masks
        image = img[:, ...]
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        if self.transform:
            # Save the randomized state so that the same transforms will be applied to the mask and image
            # when finer transforms are used
            state = torch.get_rng_state()
            torch.set_rng_state(state)

            # Data augmentation changes
            image = self.transform(image)
            mask = self.transform(mask)

        # Return image and mask
        if self.test_flag:
            return image, mask, file_name
        return image, mask

    def __len__(self) -> int:
        return len(self.filenames)
