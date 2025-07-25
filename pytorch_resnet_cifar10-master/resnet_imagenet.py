'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from torchvision.models import resnet34, resnet50, resnet101, resnet152
from torchvision.models import mobilenet_v2
from torchvision.models import densenet121
from torchvision.models import convnext_tiny

from ImageNet_dali import get_imagenet_iter_dali


__all__ = ['convnextt_ori', 'convnextt_MW', 'mobilenetv2_ori', 'mobilenetv2_MW', 'resnet152_ori', 'resnet152_MW', 'resnet101_ori', 'resnet101_MW', 'resnet34_ori', 'resnet34_MW', 'resnet50_ori', 'resnet50_MW', 'resnet50_emd0_005','resnet50_emd0_01','resnet50_emd0_02','resnet50_emd0_05', 'resnet56_emd0_1']

def convnextt_ori():
    model = convnext_tiny(weights=None)
    return model

def convnextt_MW():
    model = convnext_tiny(weights=None)
    del model.features[7][2]
    del model.features[7][1]
    del model.features[5][8]
    del model.features[5][7]
    del model.features[5][6]
    del model.features[5][5]
    del model.features[5][4]
    del model.features[5][3]
    del model.features[5][0]
    del model.features[3][0]
    return model

def densenet121_ori():
    model = densenet121(weights=None)
    return model

def densenet121_MW():
    model = densenet121(weights=None)
    return model

def mobilenetv2_ori():
    model = mobilenet_v2(weights=None)
    return model

def mobilenetv2_MW():
    model = mobilenet_v2(weights=None)
    del model.features[16]
    del model.features[15]
    del model.features[13]
    del model.features[12]
    del model.features[10]
    del model.features[9]
    del model.features[8]
    del model.features[6]
    del model.features[5]
    del model.features[3]
    return model

def resnet152_ori():
    model = resnet152(weights=None)
    return model

def resnet152_MW():
    model = resnet152(weights=None)
    del model.layer3[24]
    del model.layer3[17]
    del model.layer3[12]
    del model.layer3[10]
    del model.layer2[7]
    del model.layer2[4]
    return model

def resnet101_ori():
    model = resnet101(weights=None)
    return model

def resnet101_MW():
    model = resnet101(weights=None)
    del model.layer3[14]
    del model.layer3[4]
    del model.layer3[3]
    del model.layer2[3]
    return model

def resnet34_ori():
    model = resnet34(weights=None)
    return model

def resnet34_MW():
    model = resnet34(weights=None)
    del model.layer3[1]
    return model

def resnet50_ori():
    model = resnet50(weights=None)
    return model


def resnet50_MW():
    model = resnet50(weights=None)
    
    del model.layer2[3]

    return model

def resnet50_emd0_005():
    model = resnet50(weights=None)

    del model.layer3[1]
    del model.layer2[3]

    return model

def resnet50_emd0_01():
    model = resnet50(weights=None)

    for i in range(3, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.layer3[i]
    for i in range(3, 1, -1):  # 逆向刪除，避免索引錯亂
        del model.layer2[i]

    return model

def resnet50_emd0_02():
    model = resnet50(weights=None)

    for i in range(5, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.layer3[i]
    for i in range(3, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.layer2[i]
    for i in range(2, 1, -1):  # 逆向刪除，避免索引錯亂
        del model.layer1[i]

    return model

def resnet50_emd0_05():
    model = resnet50(weights=None)

    for i in range(5, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.layer3[i]
    for i in range(3, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.layer2[i]
    for i in range(2, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.layer1[i]

    return model

def resnet50_emd0_1():
    model = resnet50(weights=None)

    del model.layer4[1]
    for i in range(5, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.layer3[i]
    for i in range(3, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.layer2[i]
    for i in range(2, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.layer1[i]

    return model


import torch
import numpy as np
from thop import profile

def test(net, input_size=(1, 3, 224, 224)):  # CIFAR-10 用 (batch=1, 3通道, 32x32)
    total_params = 0

    # 計算參數數量
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.cpu().numpy().shape)  # 確保在 CPU 運行
    
    print("Total number of params:", total_params)
    print("Total layers:", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))

    # 計算 FLOPs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)  # 將模型移到 GPU
    input_tensor = torch.randn(input_size).to(device)

    flops, _ = profile(net, inputs=(input_tensor,), verbose=False)
    print(f"Total FLOPs: {flops:,}")  # 格式化輸出 FLOPs


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet34'):
            print(net_name)
            test(globals()[net_name]())  # 創建 ResNet 模型並測試
            print()
