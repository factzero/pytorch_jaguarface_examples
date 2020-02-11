# -*- coding: UTF-8 -*-
import torch
from config.config import cfg
from core.resnet import resnet18


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    if cfg['net'] == 'resnet18':
        net = resnet18()
    device = torch.device("cuda" if cfg['gpu'] else "cpu")
    net.to(device)
    net.eval()

    output_onnx = 'resetnet_face.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input"]
    output_names = ["output"]
    inputs = torch.randn(1, 3, 128, 128).to(device)

    torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                                   input_names=input_names, output_names=output_names)
