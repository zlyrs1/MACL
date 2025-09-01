import math
import torch.nn as nn
from modelTool.extractor import E_SPA, E_SPEC
from modelTool.mlp_head import MLPHead


# stage1：预训练网络
class ContrastNet(nn.Module):
    def __init__(self, args, mode):
        super().__init__()
        if mode == 0:
            self.encoder = E_SPA(args)
            self.projector = MLPHead(args.en_spa_channels[-1], args.mlp_hidden_size, args.projection_size)
        else:
            self.encoder = E_SPEC(args)
            self.projector = MLPHead(args.en_spec_channels[-1], args.mlp_hidden_size, args.projection_size)

    def forward(self, x):
        feature = self.encoder(x)
        projection = self.projector(feature)
        return feature, projection


# MAFF模块
class MAFF(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()

        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        # 基于1*1卷积学习通道之间的信息
        self.conv1d_1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.bn_1 = nn.BatchNorm1d(channels)
        self.act_1 = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.bn_2 = nn.BatchNorm1d(channels)
        self.act_2 = nn.ReLU(inplace=True)

        self.bn_3 = nn.BatchNorm1d(channels)
        self.act_3 = nn.Sigmoid()

    def forward(self, x):
        v = self.act_1(self.bn_1(self.conv1d_1(x.transpose(-1, -2)).transpose(-1, -2)))
        v = self.act_2(self.bn_2(self.conv1d_2(v.transpose(-1, -2)).transpose(-1, -2)))
        weight = self.act_3(self.bn_3(x + v))
        return x * weight


class Classifier(nn.Module):
    def __init__(self, fea_dims, class_num):
        super(Classifier, self).__init__()

        self.f_fusion = MAFF(fea_dims)
        self.FC = nn.Linear(fea_dims, class_num)

    def forward(self, fea):
        # MAFF
        out = self.f_fusion(fea)

        # FC
        out = out.view(out.size(0), -1)
        output = self.FC(out)

        return output
