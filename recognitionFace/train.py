# -*- coding: UTF-8 -*-
import os
import argparse
import cv2
import torch
import torch.optim as optim
from tqdm import tqdm
from config.config import cfg
from core.resnet import resnet18
from core.metrics import Arcface
from utils.data_gen import get_train_loader

parser = argparse.ArgumentParser(description='recognition face Training')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
args = parser.parse_args()


def train():
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    with torch.no_grad():
        loader, class_num = get_train_loader(cfg)

    max_epoch = cfg['epoch']

    device = torch.device("cuda" if cfg['gpu'] else "cpu")
    if cfg['network'] == 'resnet18':
        net = resnet18(cfg['embedding_size'])
    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    metric_fc = Arcface(embedding_size=cfg['embedding_size'], classnum=class_num).to(device)
    optimizer = optim.SGD([{'params': net.parameters()}, {'params': metric_fc.parameters()}],
                          lr=cfg['initial_lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])


    net.train()
    for epoch in range(max_epoch):
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
            torch.save(net.state_dict(), save_folder + cfg['name'] + '_epoch_' + str(epoch) + '_' + str(time.time()) + '.pth')


if __name__ == "__main__":
    train()
    

    image_path = "./data/87022.jpg"
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img_raw)
    img_1 = cv2.merge([r, g, b])
    #cv2.imshow('ori', img_raw)
    cv2.imshow('img_1', img_1)
    cv2.waitKey(0)
    
