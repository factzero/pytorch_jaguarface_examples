# -*- coding: UTF-8 -*-
import os
import argparse
import math
import time
import datetime
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from core.config import cfg_mnet
from core.retinaface import RetinaFace
from core.multibox_loss import MultiBoxLoss
from utils.prior_box import PriorBox
from utils.wider_face import WiderFaceDetection, detection_collate
from utils.data_augment import preproc


parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobilenet0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
args = parser.parse_args()


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size, initial_lr):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def resume_net_param(net, trained_model):
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

    return net


def train():
    cfg = None
    if args.network == "mobilenet0.25":
        cfg = cfg_mnet
    rgb_mean = (104, 117, 123)  # bgr order
    num_classes = 2
    img_dim = cfg['image_size']
    using_gpu = cfg['gpu_train']
    num_gpu = cfg['ngpu']
    batch_size = cfg['batch_size']
    max_epoch = cfg['epoch']
    num_workers = args.num_workers
    momentum = args.momentum
    weight_decay = args.weight_decay
    initial_lr = args.lr
    gamma = args.gamma
    training_dataset = args.training_dataset
    save_folder = args.save_folder
    
    net = RetinaFace(cfg=cfg, phase='test')

    if args.resume_net is not None:
        print('Loading resume network...')
        net = resume_net_param(net, args.resume_net)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
    
    if using_gpu:
        priors = priors.cuda()
        net = net.cuda()
        cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

    net.train()
    print('Loading Dataset...')
    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))
    
    epoch = 0 + args.resume_epoch
    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size
    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name'] + '_epoch_' + str(epoch) + '_' + str(time.time()) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size, initial_lr)

        # load train data
        images, targets = next(batch_iterator)
        if using_gpu:
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets, using_gpu)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                      epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')


if __name__ == "__main__":
    train()
