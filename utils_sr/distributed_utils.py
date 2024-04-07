import os
import time
import errno
import torch
import datetime
import torch.nn.functional as F
import torch.distributed as dist

from collections import defaultdict, deque
from .dice_coefficient_loss import multiclass_dice_coefficient, build_target, multiclass_dice_coefficient_mult


class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):

        """
        Track a series of values and provide access to smoothed values over a
        window or the global series average.
        """

        #  初始化默认值
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"

        # 超参数设置
        self.fmt = fmt  #
        self.count = 0  # 记录累计个数的总和
        self.total = 0.0  # 记录累计数值的总和
        self.deque = deque(maxlen=window_size)  # 利用队列来获取的数值

    def update(self, value, n=1):
        # update: 更新数值
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):

        """
        synchronize_between_processes: 同步进程数值
        Warning: does not synchronize the deque!
        dist.barrier(): 阻塞进程，等待所有进程完成计算
        dist.all_reduce(): 把所有节点上计算好的数值进行累加，然后传递给所有的节点
        """

        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg,
                               global_avg=self.global_avg, max=self.max, value=self.value)


class ConfusionMatrix(object):
    def __init__(self, num_classes):

        # 混淆矩阵 - 超参数设置
        self.mat = None
        self.num_classes = num_classes

    def update(self, a, b):
        # 获取类别数目
        n = self.num_classes

        # 创建混淆矩阵
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)

        with torch.no_grad():
            # 寻找 GT 中为目标的像素索引
            k = (a >= 0) & (a < n)

            # 统计像素真实类别 a[k] 被预测成类别 b[k] 的个数(这里的做法很巧妙)
            ind = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(ind, minlength=n ** 2).reshape(n, n)

    def reset(self):
        # 初始化函数
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):

        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()

        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)

        # 计算每个类别预测与真实目标的 iou
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        return acc_global, acc, iou

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iou = self.compute()
        '''
        记录全局正确率，每一行正确率，交并比，平均交并比
        '''
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
            acc_global.item() * 100,
            ['{:.3f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.3f}'.format(i) for i in (iou * 100).tolist()],
            iou.mean().item() * 100)


# Dice 系数计算
class DiceCoefficient(object):
    def __init__(self, num_classes: int = 2, ignore_index: int = -100):

        # 参数设置
        self.cumulative_dice = None
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.count = None

    def update(self, prediction, target):

        # 初始化 Dice 系数 和 计数器
        if self.cumulative_dice is None:
            self.cumulative_dice = torch.zeros(1, dtype=prediction.dtype, device=prediction.device)
        if self.count is None:
            self.count = torch.zeros(1, dtype=prediction.dtype, device=prediction.device)

        # 进行 One-hot 编码 和 目标编码
        prediction = F.one_hot(prediction.argmax(dim=1), self.num_classes).permute(0, 3, 1, 2).float()
        dice_target = build_target(target, self.num_classes, self.ignore_index)

        # 忽略背景，计算 Dice 系数
        self.cumulative_dice += multiclass_dice_coefficient(prediction[:, 1:], dice_target[:, 1:],
                                                            ignore_index=self.ignore_index)

        # 更新计数器
        self.count += 1

    @property
    def value(self):

        # 计算平均 Dice 系数
        if self.count == 0:
            return 0
        else:
            return self.cumulative_dice / self.count

    def reset(self):
        if self.cumulative_dice is not None:
            self.cumulative_dice.zero_()

        if self.count is not None:
            self.count.zeros_()

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.cumulative_dice)
        torch.distributed.all_reduce(self.count)


# Dice 系数计算
class DiceCoefficientMult(object):
    def __init__(self, num_classes: int = 2, ignore_index: int = -100):

        # 参数设置
        self.cumulative_dice = None
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.count = None

    def update(self, prediction, target):

        # 初始化 Dice 系数 和 计数器
        if self.cumulative_dice is None:
            self.cumulative_dice = torch.zeros(self.num_classes - 1, dtype=prediction.dtype, device=prediction.device)
        if self.count is None:
            self.count = torch.zeros(1, dtype=prediction.dtype, device=prediction.device)

        # 进行 One-hot 编码 和 目标编码
        prediction = F.one_hot(prediction.argmax(dim=1), self.num_classes).permute(0, 3, 1, 2).float()
        dice_target = build_target(target, self.num_classes, self.ignore_index)

        # 忽略背景，计算 Dice 系数
        self.cumulative_dice += multiclass_dice_coefficient_mult(prediction[:, 1:], dice_target[:, 1:],
                                                                 ignore_index=self.ignore_index)

        # 更新计数器
        self.count += 1

    @property
    def value(self):

        # 计算平均 Dice 系数
        if self.count == 0:
            return 0
        else:
            return self.cumulative_dice / self.count

    def reset(self):
        if self.cumulative_dice is not None:
            self.cumulative_dice.zero_()

        if self.count is not None:
            self.count.zeros_()

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.cumulative_dice)
        torch.distributed.all_reduce(self.count)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        # 根据 GPU 是否能够启用进行判断。
        if torch.cuda.is_available():
            '''
                命令行输出信息:迭代次数，预计该Epoch的截止时间，损失，学习率，运行时间，内存
            '''
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'])

        mb = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / mb))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)
