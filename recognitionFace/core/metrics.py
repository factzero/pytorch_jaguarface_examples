# -*- coding: UTF-8 -*-
import math
import torch
import torch.nn as nn


def l2_norm(input, axis=1):
    '''
    norm(x)=sqrt(x1^2 + x2^2 + ... + xn^2)
    x = x/norm(x)
    '''
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Arcface(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.5, easy_margin=False):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)  # initial kernel
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.m = m  # the margin value, default is 0.5
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
        self.easy_margin = easy_margin

    def forward(self, embbedings, labels):
        batch_size = embbedings.size(0)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # 计算cos(theta+m)=cos(theta).cos(m)-sin(theta).sin(m)
        cos_theta = torch.mm(embbedings, kernel_norm).clamp(-1, 1)
        sin_theta = torch.sqrt(1 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta*self.cos_m - sin_theta*self.sin_m
        # 为了使cos(theta+m)单调，cos_theta需要满足一定条件
        # easy_margin 为真：0作为阈值，直接截断
        #             为假：cos(pi-m)作为阈值，
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            cos_theta_m = torch.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm)
        # 标签对应的值为cos_theta_m，否则cos_theta
        idx = torch.arange(0, batch_size, dtype=torch.long)
        output = cos_theta * 1.0 # 防止出现inplace操作，导致不可求导
        output[idx, labels] = cos_theta_m[idx, labels]
        output = output * self.s  # scale up in order to make softmax work, first introduced in normface
        return output
