import os
import random
import importlib
import torch
import torch.backends.cudnn as cudnn
import numpy as np


# 初始化
# 1）加载数据集相关参数args
# 2）默认在指定的gpu 0上运行该程序
# 3）设置随机数
def initialize(dataset_name):
    labels_text, ignored_label = get_dataset_info(dataset_name)

    cfg_path = '{}.{}'.format('config', dataset_name)
    args = importlib.import_module(name=cfg_path)

    device = get_device()

    seed_everything(args.seed)

    return args, device, labels_text, ignored_label


def get_dataset_info(dataset_name):
    if dataset_name == "Muufl":
        labels_text = ['Trees', 'Mostly grass', 'Mixed ground surface',
                       'Dirt and sand', 'Road', 'Water',
                       'Building shadow', 'Buildings', 'Sidewalk',
                       'Yellow curb', 'Cloth panels']
        ignored_label = -1

    elif dataset_name == "Trento":
        labels_text = ['Apple trees', 'Buildings', 'Ground', 'Woods', 'Vineyard', 'Roads']
        ignored_label = -1

    elif dataset_name == "Houston":
        labels_text = ['Health grass', 'Stressed grass', 'Synthetic grass',
                       'Trees', 'Soil', 'Water', 'Residential',
                       'Commercial', 'Road', 'Highway',
                       'Railway', 'Parking lot 1', 'Parking lot 2',
                       'Tennis court', 'Running track']
        ignored_label = -1

    else:
        raise ValueError("Dataset must be one of MUUFL, Trento, Houston")
    return labels_text, ignored_label


def get_device(ordinal=0):
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        device = torch.device('cuda')
        print("Computation on CUDA GPU device".format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def seed_everything(seed: int):
    r"""
        deterministic, possibly at the cost of reduced performance
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 可选项
    cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    cudnn.benchmark = False
