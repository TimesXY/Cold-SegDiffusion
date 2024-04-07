import torch
import torch.nn.functional as F


# 对标签进行编码和填充
def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = 255):
    # 获取真实标签
    dice_target = target.clone()

    if ignore_index >= 0:

        # 将不感兴趣区域的 标签 设置为 0
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0

        # 进行 One-hot 编码，将编码后不感兴趣区域填充为 ignore_index
        dice_target = F.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        # 进行 One-hot 编码
        dice_target = F.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


# 计算 Batch 中，某个类别的 Dice 损失
def dice_coefficient(x: torch.Tensor, target: torch.Tensor, ignore_index: int = 255, epsilon=1e-6):
    # 参数设置 (误差累计，批量数目)
    dice = 0.
    batch_size = x.shape[0]

    # 循环遍历各图像，计算损失
    for i in range(batch_size):

        # 对图像和标签进行展平操作
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)

        # 找出 mask 中不为 ignore_index 的区域
        if ignore_index >= 0:
            roi_mask = torch.ne(t_i, ignore_index)

            # 获取感兴趣区域的图像和标签
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]

        # 对向量进行内积操作，得到损失
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)

        # 若分母为 0，则表明预测结果正确
        if sets_sum == 0:
            sets_sum = 2 * inter

        # 总体损失计算
        dice += (2 * inter + epsilon) / (sets_sum + epsilon)

    return dice / batch_size


# 多类别 Dice 损失计算
def multiclass_dice_coefficient(predict_target: torch.Tensor, target: torch.Tensor,
                                ignore_index: int = 255, epsilon=1e-6):
    # 误差累计参数
    dice = 0.

    # 循环遍历不同通道下的 Dice 损失，并求取均值
    for channel in range(predict_target.shape[1]):
        dice += dice_coefficient(predict_target[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / predict_target.shape[1]


# 多类别 Dice 损失计算
def multiclass_dice_coefficient_mult(predict_target: torch.Tensor, target: torch.Tensor,
                                     ignore_index: int = -100, epsilon=1e-6):
    # 误差累计参数
    dice = torch.zeros(target.shape[1], device=predict_target.device)

    # 循环遍历不同通道下的 Dice 损失
    for channel in range(predict_target.shape[1]):
        dice[channel] = dice_coefficient(predict_target[:, channel, ...],
                                         target[:, channel, ...], ignore_index, epsilon)

    return dice


# 计算 Dice 损失
def dice_loss(predict_target: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = 255):
    # Softmax 归一化处理
    predict_target = F.softmax(predict_target, dim=1)

    # 根据类别数目，选择不同损失函数
    if multiclass:
        loss = 1 - multiclass_dice_coefficient(predict_target, target, ignore_index=ignore_index)
    else:
        loss = 1 - dice_coefficient(predict_target, target, ignore_index=ignore_index)

    return loss
