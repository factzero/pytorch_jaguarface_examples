# -*- coding: UTF-8 -*-
import torch
import numpy as np


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    # 将(cx, cy, w, h) 形式的box坐标转换成 (xmin, ymin, xmax, ymax) 形式
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
        box_a 与 box_b 两个集合中任意两个 box 的交集, 
        其中res[i][j]代表box_a中第i个box与box_b中第j个box的交集.(非对称矩阵)
    """
    # 先将两个box的维度扩展至相同维度: [num_obj, num_priors, 4], 然后计算面积的交集
    A = box_a.size(0)
    B = box_b.size(0)
    # box_a 左上角/右下角坐标 expand以后, 维度会变成(A,B,2), 其中, 具体可看 expand 的相关原理. 
    #       box_b也是同理, 这样做是为了得到a中某个box与b中某个box的左上角(min_xy)的较大者(max)
    # unsqueeze 为增加维度的数量, expand 为扩展维度的大小
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    # 高×宽, 返回交集的面积, shape 刚好为 [A, B]
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    # 任意两个box的交集面积, shape为[A, B], 即[num_obj, num_priors]
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    # [A, B], 返回任意两个box之间的交并比, res[i][j] 代表box_a中的第i个box与box_b中的第j个box之间的交并比.
    return inter / union  # [A,B]


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        landms: (tensor) Ground truth landms, Shape [num_obj, 10].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        landm_t: (tensor) Tensor to be filled w/ endcoded landm targets.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landm preds.
    """
    """
    threshold: (float) 确定是否匹配的交并比阈值
    truths: (tensor: [num_obj, 4]) 存储真实 box 的边框坐标
    priors: (tensor: [num_priors, 4], 存储推荐框的坐标, 注意, 此时的框是 default box
    variances: cfg['variance'], [0.1, 0.2], 用于将坐标转换成方便训练的形式(参考RCNN系列对边框坐标的处理)
    labels: (tensor: [num_obj]), 代表了每个真实 box 对应的类别的编号
    """
    # jaccard index
    overlaps = jaccard(truths, point_form(priors))

    # (Bipartite Matching)
    # [num_objects, 1] best prior for each ground truth
    #  得到对于每个 gt box 来说的匹配度最高的 prior box, 前者存储交并比, 后者存储prior box在num_priors中的位置
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    # 得到对于每个 prior box 来说的匹配度最高的 gt box
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    # 维度压缩
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior
    # 该语句会将与gt box匹配度最好的prior box 的交并比置为 2, 确保其最大, 以免防止某些 gtbox 没有匹配的 priorbox.

    # 假想一种极端情况, 所有的priorbox与某个gtbox(标记为G)的交并比为1, 而其他gtbox分别有一个交并比
    # 最高的priorbox, 但是肯定小于1(因为其他的gtbox与G的交并比肯定小于1), 这样一来, 就会使得所有
    # 的priorbox都与G匹配, 为了防止这种情况, 我们将那些对gtbox来说, 具有最高交并比的priorbox,
    # 强制进行互相匹配, 即令best_truth_idx[best_prior_idx[j]] = j, 详细见下面的for循环

    # 注意!!: 因为 gt box 的数量要远远少于 prior box 的数量, 因此, 同一个 gt box 会与多个 prior box 匹配.
    
    
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):     # 判别此anchor是预测哪一个boxes
        best_truth_idx[best_prior_idx[j]] = j
        # best_prior_idx[j] 代表与第j个gt box交并比最高的 prior box 的下标,
        # 将与该gtbox匹配度最好的 prior box 的下标改为j, 由此,完成了该 gtbox 与第best_prior_idx[j]个 prior box 的匹配.
        # 这里的循环只会进行num_obj次, 剩余的匹配为 best_truth_idx 中原本的值.
        # 这里处理的情况是：
        #   priorbox中第i个box与gtbox中第k个box的交并比最高, 即 best_truth_idx[i]= k
        #   但对best_prior_idx[k]来说, 它却与priorbox的第l个box有最高的交并比, 即best_prior_idx[k]=l
        #   而对gtbox的另一个边框gtbox[j]来说, 它与priorbox[i]的交并比最大, 即best_prior_idx[j] = i.
        #   那么, 我们就应该将best_truth_idx[i]=k 修改成best_truth_idx[i]=j. 即令priorbox[i]与gtbox[j]对应.
        # 这样做的原因: 防止某个gtbox没有匹配的 prior box.

    # truths 的shape 为[num_objs, 4], 而best_truth_idx是一个指示下标的列表
    # 会返回一个shape为 [num_priors, 4], 代表的就是与每个priorbox匹配的gtbox的坐标值.
    matches = truths[best_truth_idx]            # Shape: [num_priors,4] 此处为每一个anchor对应的bbox取出来
    conf = labels[best_truth_idx]               # Shape: [num_priors]   此处为每一个anchor对应的label取出来
    conf[best_truth_overlap < threshold] = 0    # label as background   overlap<0.35的全部作为负样本
    loc = encode(matches, priors, variances)    # 返回编码后的中心坐标和宽高.

    matches_landm = landms[best_truth_idx]
    landm = encode_landm(matches_landm, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    landm_t[idx] = landm


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    # 对边框坐标进行编码, 需要宽度方差和高度方差两个参数, 具体公式可以参见原文公式
    # matched: [num_priors,4] 存储的是与priorbox匹配的gtbox的坐标. 形式为(xmin, ymin, xmax, ymax)
    # priors: [num_priors, 4] 存储的是priorbox的坐标. 形式为(cx, cy, w, h)

    # dist b/t match center and prior's center
    # 用互相匹配的gtbox的中心坐标减去priorbox的中心坐标, 获得中心坐标的偏移量
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # 令中心坐标分别除以 d_i^w 和 d_i^h, 正如原文公式所示，参考SSD
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    # 令互相匹配的gtbox的宽高除以priorbox的宽高.
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def encode_landm(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 10].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded landm (tensor), Shape: [num_priors, 10]
    """

    # dist b/t match center and prior's center
    matched = torch.reshape(matched, (matched.size(0), 5, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    # g_cxcy /= priors[:, :, 2:]
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    # return target for smooth_l1_loss
    return g_cxcy


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
