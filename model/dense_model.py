from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self, config):
        super(DenseNet,self).__init__()
        self.densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.conv     = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        )
        self.avgpool    = nn.AvgPool2d((7,7))
        self.classifier = nn.Linear(64, config.CLASS_N)
        self.softmax    = nn.Softmax()
        
    def forward(self, x, seq, annotations, train=True):
        features = self.densenet.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.avgpool(out)
        out = out.view(out.shape[0],out.shape[1])
        out = self.classifier(out)
        out = self.softmax(out)