# -*- coding: UTF-8 -*-
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import mxnet as mx
from tqdm import tqdm
import os

    
def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def load_mx_rec(rec_path):
    save_path = os.path.join(rec_path, 'imgs')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    idx_path = os.path.join(rec_path, 'train.idx')
    uri_path = os.path.join(rec_path, 'train.rec')
    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, uri_path, 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = Image.fromarray(img)
        label_path = os.path.join(save_path, str(label))
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        img_name = '{}.jpg'.format(idx)
        img.save(os.path.join(label_path, img_name), quality=95)

