import numpy as np
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

class ResNet(nn.Module):
    '''
    inputs : (B x N x 295)
    outputs : (B x 512 x 1)
    '''
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(config.NUM_FEATURES, 64, kernel_size=3, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv2 = get_residual_layer(config.RESNET_DEPTH, 64, 128, 3)
        self.conv3 = get_residual_layer(config.RESNET_DEPTH, 128, 256, 3)
        self.conv4 = get_residual_layer(config.RESNET_DEPTH, 256, config.D_MODEL, 3)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, csv_features, *args, **kwargs):
        outputs = self.conv1(csv_features)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.avg_pool(outputs)
        return outputs


def get_residual_layer(num_layer, in_channels, out_channels, kernel_size):
    layers = nn.Sequential()
    for i in range(num_layer):
        parameters = [out_channels, out_channels, kernel_size] if i else [in_channels, out_channels, kernel_size]
        layer = ResidualCell(*parameters)
        layers.add_module(f'{i}', layer)
    return layers

    
class ResidualCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        if in_channels==out_channels:
            self.shortcut = nn.Identity(stride=2)
            stride = 1
        else:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)
            stride = 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels,out_channels,kernel_size, stride=stride, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels,out_channels,kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        Fx = self.conv(x)
        x = self.shortcut(x)
        return Fx + x