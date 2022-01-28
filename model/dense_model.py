import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import transforms
import numpy as np

class DenseNet(nn.Module):
    def __init__(self, config):
        super(DenseNet,self).__init__()
        self.densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        for param in self.densenet.parameters():
            param.requires_grad = False
        self.conv     = nn.Sequential(
            SimpleConv2d(1024, 2048, kernel_size=3, padding=1, bias=False),
            SimpleConv2d(2048, 2048, kernel_size=3, padding=1, bias=False),
            SimpleConv2d(2048, config.CLASS_N, kernel_size=3, padding=1, bias=False),
        )
        # self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        # self.inception.fc = nn.Linear(self.inception.fc.in_features, 512)
        # for name, param in self.inception.named_parameters():
        #     if 'fc.weight' in name or 'fc.bias' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        
    def forward(self, img, seq, labels=None, train=True, **kwargs):
        # resize
        # img = transforms.Resize(224)(img)
        
        # densenet
        features = self.densenet.features(img)
        out = F.relu(features, inplace=True)
        
        out = self.conv(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.shape[0], -1)
        
        return out
   
class SimpleConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, **kwargs)
        self.bn   = nn.BatchNorm2d(output_dim)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x