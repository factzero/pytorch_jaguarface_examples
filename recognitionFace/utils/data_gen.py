# -*- coding: UTF-8 -*-
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

def get_train_dataset(imgs_folder):
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num


def get_train_loader(conf, data_folder):
    if conf['data_mode'] == 'emore':
        ds, class_num = get_train_dataset(data_folder)
    loader = DataLoader(ds, batch_size=conf['batch_size'], shuffle=True, 
                        pin_memory=conf['pin_memory'], num_workers=conf['num_workers'])
    return loader, class_num
