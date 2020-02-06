# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets, using_gpu):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, landm_data = predictions
        priors = priors
        num = loc_data.size(0)           # num = batch_size
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :4].data     # [num_objs, 4]
            labels = targets[idx][:, -1].data     # [num_objs]
            landms = targets[idx][:, 4:14].data   # [num_objs, 10]
            defaults = priors.data
            # 关键函数, 实现候选框与真实框之间的匹配
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
        
        zeros = torch.tensor(0)
        if using_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()
            zeros = zeros.cuda()

        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros          # 筛选出 >0 的box下标(大部分都是=0的)
        # 求和, 取得满足条件的box的数量
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining        
        batch_conf = conf_data.view(-1, self.num_classes)
        # conf_t: [batch, num_priors]
        # loss_c: [batch*num_priors, 1], 计算每个priorbox预测后的损失
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        # 难负样本挖掘, 按照loss进行排序, 取loss最大的负样本参与更新
        # 将所有的pos下标的box的loss置为0(pos指示的是正样本的下标)
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        # 将 loss_c 的shape 从 [batch*num_priors, 1] 转换成 [batch, num_priors]
        loss_c = loss_c.view(num, -1)
        # 进行降序排序, 并获取到排序的下标
        _, loss_idx = loss_c.sort(1, descending=True)
        # 将下标进行升序排序, 并获取到下标的下标
        _, idx_rank = loss_idx.sort(1)
        # num_pos: [batch, 1], 统计每个样本中的obj个数
        num_pos = pos.long().sum(1, keepdim=True)
        # 根据obj的个数, 确定负样本的个数(正样本的3倍)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # 获取到负样本的下标
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        # 计算包括正样本和负样本的置信度损失
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # 按照pos_idx和neg_idx指示的下标筛选参与计算损失的预测数据
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        # 将损失函数归一化后返回
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm
