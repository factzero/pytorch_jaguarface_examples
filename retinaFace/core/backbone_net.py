# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3sn_bn(in_c, out_c, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv3x3sn_bn_no_relu(in_c, out_c, stride):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_c),
    )


def conv1x1sn_bn(in_c, out_c, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_dw(in_c, out_c, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
        nn.BatchNorm2d(in_c),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),

        nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


class SSH(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSH, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.conv3X3 = conv3x3sn_bn_no_relu(in_channels, out_channels//2, stride=1)
        self.conv5X5_1 = conv3x3sn_bn(in_channels, out_channels//4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv3x3sn_bn_no_relu(out_channels//4, out_channels//4, stride=1)
        self.conv7X7_2 = conv3x3sn_bn(out_channels//4, out_channels//4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv3x3sn_bn_no_relu(out_channels//4, out_channels//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)
        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv1x1sn_bn(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv1x1sn_bn(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv1x1sn_bn(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        self.merge1 = conv3x3sn_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv3x3sn_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input):
        input = list(input.values())
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv3x3sn_bn(3, 8, 2, leaky=0.1),
            conv_dw(8, 16, 1),
            conv_dw(16, 32, 2),
            conv_dw(32, 32, 1),
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

