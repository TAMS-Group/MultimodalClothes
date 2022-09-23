#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from data_utils.MultiClothesDataLoader import MultiClothesDataLoader
from typing import Iterator, Dict

def get_model(num_class, config_dict):
    return ResNet(num_class, config_dict['use_cpu'])

class ResNet(nn.Module):
    def __init__(self, num_class, use_cpu):
        super(ResNet, self).__init__()
        self.model_ft = models.resnet18(pretrained=True)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(self.num_ftrs, num_class)
        self.use_cpu = use_cpu

    def apply_transformations(self, input_tensor):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(input_tensor)

    def forward(self, data: Dict[MultiClothesDataLoader.Modalities, torch.Tensor]):
        x: torch.Tensor = data[MultiClothesDataLoader.Modalities.RGB].float() / 255.0
        x = x.transpose(1, 3)
        x = x.transpose(2, 3)
        x = self.apply_transformations(x)
        if not self.use_cpu:
            x = x.cuda()
        x = self.model_ft(x)
        x = F.log_softmax(x, dim=1)
        return x, None

    def parameters(self, recurse: bool = True) -> Iterator:
        return self.model_ft.parameters(recurse)

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        loss = torch.nn.functional.nll_loss(pred, target)

        return loss