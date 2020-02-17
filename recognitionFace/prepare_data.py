# -*- coding: UTF-8 -*-
from utils.data_pipe_mxnet import load_bin, load_mx_rec
import argparse

parser = argparse.ArgumentParser(description='Prepare Data')
parser.add_argument('--data_folder', default='./data/faces_emore/', help='Training dataset directory')
args = parser.parse_args()

if __name__ == '__main__':
    print("prepare data:{}".format(args.data_folder))
    load_mx_rec(args.data_folder)
