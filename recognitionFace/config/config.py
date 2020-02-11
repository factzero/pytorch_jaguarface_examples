# -*- coding: UTF-8 -*-

cfg = {
    'name': 'recognition face ',
    'network': 'resnet18',
    'gpu': True,
    'data_root': 'D:/80dataset',
    'data_folder': 'D:/80dataset/faces_emore/imgs/',
    'data_mode': 'emore',
    'epoch': 250,
    'batch_size': 8,
    'pin_memory': True,
    'num_workers': 3,
    'optimizer': 'sgd',
    'initial_lr': 1e-1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'embedding_size': 512
}
