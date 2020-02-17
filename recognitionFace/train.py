# -*- coding: UTF-8 -*-
import os
import argparse
import cv2
import time
import torch
import torch.optim as optim
from tqdm import tqdm
from config.config import cfg
from core.resnet import resnet18
from core.metrics import Arcface
from utils.data_gen import get_train_loader


parser = argparse.ArgumentParser(description='recognition face Training')
parser.add_argument('--training_dataset', default='./data/faces_emore/imgs/', help='Training dataset directory')
parser.add_argument('--network', default='resnet18', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
args = parser.parse_args()


def resume_net_param(net, trained_model):
    print('Loading trained model: {}'.format(trained_model))
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
    save_folder = args.save_folder
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    print('load dataset ...')
    with torch.no_grad():
        loader, class_num = get_train_loader(cfg, args.training_dataset)

    print('network ...')
    device = torch.device("cuda" if cfg['gpu'] else "cpu")
    if args.network == 'resnet18':
        net = resnet18(cfg['embedding_size'])
 
    if args.resume_net is not None:
        print('Loading resume network...')
        net = resume_net_param(net, args.resume_net)
 
    net = net.to(device)

    print('init cirterion ...')
    criterion = torch.nn.CrossEntropyLoss()
    metric_fc = Arcface(embedding_size=cfg['embedding_size'], classnum=class_num).to(device)
    optimizer = optim.SGD([{'params': net.parameters()}, {'params': metric_fc.parameters()}],
                          lr=cfg['initial_lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])

    max_epoch = cfg['epoch']
    start_epoch = args.resume_epoch

    net.train()
    for epoch in range(start_epoch, max_epoch):
        print('Epoch:{}/{}, batch:{}'.format(epoch, max_epoch, cfg['batch_size']))
        for imgs, labels in tqdm(iter(loader)):
            imgs = imgs.to(device)
            labels = labels.to(device)
            # forward
            features = net(imgs)
            # backprop
            output = metric_fc(features, labels)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch % 10 == 0 and epoch > 0):
            save_w_file = args.network + '_epoch_' + str(epoch) + '_' + str(time.time()) + '.pth'
            print('saving weights : {}'.format(save_w_file))
            torch.save(net.state_dict(), save_folder + save_w_file)


if __name__ == "__main__":
    train()
    

    image_path = "./data/87022.jpg"
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img_raw)
    img_1 = cv2.merge([r, g, b])
    #cv2.imshow('ori', img_raw)
    cv2.imshow('img_1', img_1)
    cv2.waitKey(0)
    
