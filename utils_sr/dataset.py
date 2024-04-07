import os
import torch
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils.data import Dataset

# 环境设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 将高宽不同的图像拷贝至新的 Tensor 组成批量，用于训练和验证。
def cat_list(images, fill_value=0):
    # 计算该 batch 数据中，通道、高度、宽度的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # 在最大的 Tensor 维度中加入 batch 维度
    batch_shape = (len(images),) + max_size
    # 生成足够保存全部图像的新的 Tensor，并默认填充为 0
    batched_img = images[0].new(*batch_shape).fill_(fill_value)
    # 将图像拷贝至新的 Tensor 中
    for img, pad_img in zip(images, batched_img):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_img


# 自定义数据读取类 Data_Segmentation
class DataSegmentation(data.Dataset):
    def __init__(self, data_path, dataset, is_folder, transforms=None):
        super(DataSegmentation, self).__init__()

        # 读取 txt 文件
        fh_txt = open(data_path + dataset, 'r')
        images_labels = []
        for line in fh_txt:
            line = line.rstrip()  # 默认删除的是空白符 ('\n', '\r', '\t', ' ')
            images_labels.append(line)

        # 超参数的赋值和设置
        self.transform = transforms

        # 获取数据路径和训练图像路径
        self.data_path = data_path + is_folder
        self.images_labels = images_labels

        # 确保图像存在对应标签
        self.transforms = transforms

    def __getitem__(self, index):

        # 获取待读取图像名称和绝对路径
        name = self.images_labels[index] + '.jpg'
        img_path = os.path.join(self.data_path, name)

        # 获取待读取图像分割掩码和绝对路径
        mask_name = self.images_labels[index] + '_Segmentation.png'
        msk_path = os.path.join(self.data_path, mask_name)

        # 图像和掩码的格式转换
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path)
        mask = np.array(mask) / 255

        # 这里转回 PIL 的原因是，transforms中是对 PIL 数据进行处理
        target = Image.fromarray(mask)

        # 对图像进行数据增强、返回增强后的图像和标签
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images_labels)

    @staticmethod
    def collate_fn(batch):

        # 将批量中的图像和标签划分为 images, targets
        images, targets = list(zip(*batch))

        # 将高宽不同的图像拷贝至新的 Tensor 组成批量，用于训练和验证。
        batched_img = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)

        # 返回填充后的结果
        return batched_img, batched_targets


# 自定义数据读取类 Data_Segmentation
class DataSegmentationWithPath(data.Dataset):
    def __init__(self, data_path, dataset, is_folder, transforms=None):
        super(DataSegmentationWithPath, self).__init__()

        # 读取 txt 文件
        fh_txt = open(data_path + dataset, 'r')
        images_labels = []
        for line in fh_txt:
            line = line.rstrip()  # 默认删除的是空白符 ('\n', '\r', '\t', ' ')
            images_labels.append(line)

        # 超参数的赋值和设置
        self.transform = transforms

        # 获取数据路径和训练图像路径
        self.data_path = data_path + is_folder
        self.images_labels = images_labels

        # 确保图像存在对应标签
        self.transforms = transforms

    def __getitem__(self, index):

        # 获取待读取图像名称和绝对路径
        name = self.images_labels[index] + '.jpg'
        img_path = os.path.join(self.data_path, name)

        # 获取待读取图像分割掩码和绝对路径
        mask_name = self.images_labels[index] + '_Segmentation.png'
        msk_path = os.path.join(self.data_path, mask_name)

        # 图像和掩码的格式转换
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path)
        mask = np.array(mask) / 255

        # 这里转回 PIL 的原因是，transforms中是对 PIL 数据进行处理
        target = Image.fromarray(mask)

        # 对图像进行数据增强、返回增强后的图像和标签
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, mask_name

    def __len__(self):
        return len(self.images_labels)


class ISICDataset(Dataset):
    def __init__(self, data_path, dataset, is_folder, transform=None, training=True, flip_p=0.0):

        # 读取 txt 文件
        fh_txt = open(data_path + dataset, 'r')
        images_labels = []
        for line in fh_txt:
            line = line.rstrip()  # 默认删除的是空白符 ('\n', '\r', '\t', ' ')
            images_labels.append(line)

        # 超参数的赋值和设置
        self.flip_p = flip_p
        self.training = training
        self.transform = transform

        # 获取数据路径和训练图像路径
        self.data_path = data_path + is_folder
        self.images_labels = images_labels

    def __len__(self):
        return len(self.images_labels)

    def __getitem__(self, index):

        # 获取待读取图像名称和绝对路径
        name = self.images_labels[index] + '.jpg'
        img_path = os.path.join(self.data_path, name)

        # 获取待读取图像分割掩码和绝对路径
        mask_name = self.images_labels[index] + '_Segmentation.png'
        msk_path = os.path.join(self.data_path, mask_name)

        # 图像和掩码的格式转换
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # 数据增强过程
        if self.transform:
            # 保存随机状态，以便在使用更精细变换时，相同变换将应用于掩码和图像
            state = torch.get_rng_state()
            torch.set_rng_state(state)

            # 数据增强
            img = self.transform(img)
            mask = self.transform(mask)

            # 随机旋转
            if random.random() < self.flip_p:
                img = F.vflip(img)
                mask = F.vflip(mask)

        if self.training:
            return img, mask

        return img, mask, mask_name


class GenericNpyDataset(torch.utils.data.Dataset):
    def __init__(self, directory: str, transform, test_flag: bool = True):
        """ 用于加载 npy 文件的通用数据集, npy 存储 3D 数组, 通道 0 是图像, 通道 1 是标签
        """

        super().__init__()
        self.transform = transform
        self.test_flag = test_flag
        self.directory = os.path.expanduser(directory)
        self.filenames = [x for x in os.listdir(self.directory) if x.endswith('.npy')]

    def __getitem__(self, x: int):
        # 获取文件名称
        file_name = self.filenames[x]

        # 读取文件
        npy_img = np.load(os.path.join(self.directory, file_name))

        # 获取图像, 并进行维度交换 通道在前
        img = npy_img[:, :, :1]
        img = torch.from_numpy(img).permute(2, 0, 1)

        # 获取掩码，并二值化
        mask = npy_img[:, :, 1:]
        mask = np.where(mask > 0, 1, 0)

        # 图像和掩码的格式转换
        image = img[:, ...]
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        if self.transform:
            # 保存随机状态，以便在使用更精细变换时，相同变换将应用于掩码和图像
            state = torch.get_rng_state()
            torch.set_rng_state(state)

            # 数据增强变化
            image = self.transform(image)
            mask = self.transform(mask)

        # 返回图像和掩码
        if self.test_flag:
            return image, mask, file_name
        return image, mask

    def __len__(self) -> int:
        return len(self.filenames)
