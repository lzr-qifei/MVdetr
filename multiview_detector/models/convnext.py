import torch
import torch.nn as nn
import timm
import os
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

class ConvNeXt(nn.Module):
    def __init__(self):
        ckpt_path = '/root/pytorch_model.bin'
        super(ConvNeXt, self).__init__()
        self.backbone = timm.create_model('convnext_base', pretrained=True,pretrained_cfg_overlay=dict(file=ckpt_path),features_only=True)
        # self.fc = nn.Linear(self.backbone.num_features, num_classes)
        self.backbone.head = nn.Identity()  # 移除原来的全连接层

    def forward(self, x):
        # print(x.shape)
        # x1 = self.backbone(x)[0]
        # x2 = self.backbone(x)[1]
        x3 = self.backbone(x)
        # x4 = self.backbone(x)[3]
        # print(x)
        print('backbone output: ',x3.shape)
        # x = self.fc(x)
        return x3

def build_convnext():
    return ConvNeXt()
