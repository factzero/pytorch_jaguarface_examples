# -*- coding: UTF-8 -*-
import argparse
import cv2
import numpy as np
import time
import torch
from core.config import cfg_mnet
from core.retinaface import RetinaFace


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth', 
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobilenet0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--long_side', default=640, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    cfg = None
    if args.network == "mobilenet0.25":
        cfg = cfg_mnet
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # ------------------------ export -----------------------------
    output_onnx = 'retinaface.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(1, 3, args.long_side, args.long_side).to(device)

    torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                                   input_names=input_names, output_names=output_names)
