import os
import torch
import torch.nn.functional as F
from datetime import datetime, timedelta, timezone


# 获取时间戳
def get_timestamp():
    """URL: https://blog.csdn.net/weixin_39715012/article/details/121048110
    """
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )

    # 协调世界时
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    # 北京时间
    beijing_now = utc_now.astimezone(SHA_TZ)

    return beijing_now.strftime("%Y_%m_%d_%H_%M_%S")


def save_model(model, dataset_name, model_name, best_value):
    model_dir = os.path.join('./checkpoints', dataset_name, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    filename = 'best_model' + '_' + str(round(best_value, 5)) + '.pth'
    torch.save(model.state_dict(), os.path.join(model_dir, filename))


# 获取网络参数
# 优化器相关函数,通过收集对比学习模型当前的网络参数,以此来更新优化器的学习率
def collect_params(model_list, exclude_bias_and_bn=True):
    """
    exclude_bias_and bn: exclude bias and bn from both weight decay and LARS adaptation
        in the PyTorch implementation of ResNet, `downsample.1` are bn layers
    """
    param_list = []
    for model in model_list:
        for name, param in model.named_parameters():
            if exclude_bias_and_bn and ('bn' in name or 'downsample.1' in name or 'bias' in name):
                param_dict = {'params': param, 'weight_decay': 0., 'lars_exclude': True}
            else:
                param_dict = {'params': param}
            param_list.append(param_dict)
    return param_list


def adjust_learning_rate(optimizer, epoch, lr_encoder, lr):
    if epoch < 300:
        lr = lr * (0.95 ** (epoch // 100))
    else:
        lr = lr * (0.8 ** (epoch // 300))
        k = optimizer.param_groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 对比学习损失函数的计算
def regression_loss(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 2 - 2 * (x * y).sum(dim=-1)
