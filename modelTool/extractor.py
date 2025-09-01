import torch
from torch import nn
import numpy as np


def spatial_weight_matrix(fea_height, fea_width, sigma_s):
    center_x, center_y = fea_width // 2, fea_height // 2
    spatial_weights = np.zeros((fea_height, fea_width))

    for y in range(fea_height):
        for x in range(fea_width):
            spatial_weights[y, x] = spatial_weight(x, y, center_x, center_y, sigma_s)

    spatial_weights = np.asarray(spatial_weights, dtype='float32')
    spatial_weights = torch.from_numpy(spatial_weights)
    spatial_weights = spatial_weights / torch.sum(spatial_weights)
    return spatial_weights


def spatial_weight(x, y, center_x, center_y, sigma_s):
    d_space = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    weight = np.exp(- (d_space**2) / (2 * sigma_s**2))
    return weight


device = torch.device('cuda')

scale = [5, 7, 10, 21]
Weight_SPA = {}
for i in scale:
    weight_spa = spatial_weight_matrix(i, i, 0.1)
    weight_spa = weight_spa.to(device)
    Weight_SPA[str(i)] = weight_spa


class SpecAtt(nn.Module):
    def __init__(self, channels, reduction_radio=16):
        super().__init__()
        self.channels = channels
        self.inter_channels = self.channels // reduction_radio

        self.mlp = nn.Sequential(
            nn.Conv2d(self.channels, self.inter_channels,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            nn.Conv2d(self.inter_channels, self.channels,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, H, W = x.shape
        s = int(H / 2)
        center = x[:, :, s, s].unsqueeze(2).unsqueeze(3)
        # m = torch.sum(torch.sum(x * Weight_SPA[str(W)], dim=3, keepdim=True), dim=2, keepdim=True)
        center_out = self.mlp(center)
        spec_mask = self.sigmoid(center_out)
        return spec_mask


class SpaAtt(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1,
                              kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        maxpool = x.argmax(dim=1, keepdim=True)
        avgpool = x.mean(dim=1, keepdim=True)
        #   5(99.33) 10(99.19) 15(98.70)
        # b, c, h, w = avgpool.shape
        # weight = Weight_SPA[str(h)].expand(b, c, h, w) * 5
        out = torch.cat([maxpool, avgpool], dim=1)
        out = self.conv(out)
        spa_mask = self.sigmoid(out)
        return spa_mask


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CA(nn.Module):
    """
    Spatial attention module from Coordinate Attention.
    Reference: https://github.com/Andrew-Qibin/CoordAttention/blob/main/coordatt.py
    """

    def __init__(self, in_channels, out_channels, reduction=32):
        super().__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        hidden_channels = max(8, in_channels // reduction)

        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        _, _, H, W = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv(y)
        y = self.bn(y)
        y = self.act(y)

        mask_h, mask_w = torch.split(y, [H, W], dim=2)
        mask_w = mask_w.permute(0, 1, 3, 2)

        mask_h = self.conv_h(mask_h).sigmoid()
        mask_w = self.conv_w(mask_w).sigmoid()

        mask = mask_h * mask_w
        return mask


class Conv_InteChannel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return out


class Conv_InteSpa(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return out


class ShortcutProjection_spa(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ShortcutProjection_spec(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


# Spatial_Residual_Unit
class SpaRU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = ShortcutProjection_spa(in_channels, out_channels)

        self.act2 = nn.ReLU(inplace=True)

        self.avgPooling = nn.AvgPool2d(2, stride=2)

        self.spa_att = SpaAtt()
        self.spec_att = SpecAtt(out_channels)

    def forward(self, x):
        out_1 = self.act1(self.bn1(self.conv1(x)))
        out_2 = self.bn2(self.conv2(out_1))
        out_2mask = self.spa_att(out_2)
        out_2 = out_2 * out_2mask
        residual = self.shortcut(x)
        output = self.act2(residual + out_2)

        output = self.avgPooling(output)
        return output


# Spectral_Residual_Unit
class SpecRU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # stride 可以尝试设置不同的值
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = ShortcutProjection_spec(in_channels, out_channels)

        self.act2 = nn.ReLU(inplace=True)

        self.spa_att = SpaAtt()
        self.spec_att = SpecAtt(out_channels)

    def forward(self, x):
        out_1 = self.act1(self.bn1(self.conv1(x)))
        out_2 = self.bn2(self.conv2(out_1))
        out_2mask = self.spec_att(out_2)
        out_2 = out_2 * out_2mask
        residual = self.shortcut(x)
        output = self.act2(residual + out_2)
        return output


class E_SPA(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_channel = args.en_spa_input_channel
        channels = args.en_spa_channels

        """spatial attention module"""
        self.spa_att = CA(in_channels=channels[0], out_channels=channels[0])

        self.residual_unit1 = SpaRU(input_channel, channels[0])
        self.residual_unit2 = SpaRU(channels[0], channels[1])
        self.residual_unit3 = SpaRU(channels[1], channels[2])
        self.Conv_InteChannel = Conv_InteChannel(channels[2], channels[3])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.residual_unit1(x)
        out = self.residual_unit2(out)
        out = self.residual_unit3(out)
        hsi_feat = self.Conv_InteChannel(out)

        feat = self.avg_pool(hsi_feat)
        feat = self.flatten(feat)
        return feat


class E_SPEC(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_channel = args.en_spec_input_channel
        channels = args.en_spec_channels

        """spatial attention module"""
        self.spa_att = CA(in_channels=channels[0], out_channels=channels[0])

        """spectral attention module"""
        self.spec_att = SpecAtt(input_channel)

        self.residual_unit1 = SpecRU(input_channel, channels[0])
        self.residual_unit2 = SpecRU(channels[0], channels[1])
        self.residual_unit3 = SpecRU(channels[1], channels[2])
        self.Conv_InteSpa = Conv_InteSpa(channels[2], channels[3])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.residual_unit1(x)
        out = self.residual_unit2(out)
        out = self.residual_unit3(out)
        hsi_feat = self.Conv_InteSpa(out)

        feat = self.avg_pool(hsi_feat)
        feat = self.flatten(feat)
        return feat
