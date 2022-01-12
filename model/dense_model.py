from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self, config):
        super(DenseNet,self).__init__()
        self.densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.conv     = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(512, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        )
        self.classifier = nn.Linear(64, config.CLASS_N)
        
    def forward(self, x, seq, annotations, train=True):
        features = self.densenet.features(x)
        out = F.relu(features, inplace=True)
        out = self.conv(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.squeeze(out)
        out = self.classifier(out)
        out = F.softmax(out)
        
        if x.shape[0]==1:
            out = out.unsqueeze(0)
        
        return out