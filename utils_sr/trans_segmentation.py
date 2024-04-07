from ClodSegDiffusion.ColdSegDiffusion_256.utils_sr import transforms as T


class SegmentationPresetTrain:
    """ 训练数据的预处理过程 """

    def __init__(self, base_size, crop_size, h_flip_prob=0.5, v_flip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        # 获取随机放缩的图像范围 - 随机放缩
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)
        trans = [T.RandomResize(min_size, max_size)]

        # 按照概率进行随机翻转
        if h_flip_prob > 0:
            trans.append(T.RandomHorizontalFlip(h_flip_prob))
        if v_flip_prob > 0:
            trans.append(T.RandomVerticalFlip(v_flip_prob))

        # 训练数据的标准化和随机裁剪
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    """ 验证数据的预处理过程 """

    def __init__(self, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(crop_size),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, base_size=565, crop_size=480, h_flip_prob=0.5, v_flip_prob=0.5,
                  mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if train:
        src_r = SegmentationPresetTrain(base_size, crop_size, h_flip_prob=h_flip_prob, v_flip_prob=v_flip_prob,
                                        mean=mean, std=std)
    else:
        src_r = SegmentationPresetEval(crop_size, mean=mean, std=std)
    return src_r
