import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import transforms
import numpy as np

from model.vit_model import ViT, MyViT
from model.csv_encoder import *

class LAB_model(MyViT):
    def __init__(self, config):
        config.IMAGE_HEIGHT //= 4
        config.IMAGE_WIDTH  //= 4
        super().__init__(config)
        self.in_channels = 65 if config.USE_SPOT else 64
        fh, fw = self.vit.fh, self.vit.fw
        self.vit.patch_embedding = nn.Conv2d(self.in_channels, config.D_MODEL, kernel_size=(fh, fw), stride=(fh, fw))
        self.lab_model = LabExtractor(config)
        
    def forward(self, img, seq, *args, **kwargs):
        
        LAB = self.lab_model(img[:,:3].detach())

        if self.in_channels==4:
            outputs = torch.cat((LAB, img[:,3]), dim=1) #(BATCH_SIZE, 64+1, 128, 184)
        else:
            outputs = LAB
            
        outputs = super().forward(outputs, seq, *args, **kwargs)
        
        return outputs

class LabExtractor(nn.Module):
    def __init__(self, config):
        super(LabExtractor,self).__init__()
        self.L_conv = nn.Sequential(
            SimpleConv2d(1,6, kernel_size=3, padding=1, stride=2, bias=False),
            SimpleConv2d(6,6, kernel_size=3, padding=1, bias=False),
            SimpleConv2d(6,13, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(2,2)
        )
        self.AB_conv = nn.Sequential(
            SimpleConv2d(2,26, kernel_size=3, padding=1, stride=2, bias=False),
            SimpleConv2d(26,26, kernel_size=3, padding=1, bias=False),
            SimpleConv2d(26,51, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(2,2)
        )
        
    def forward(self, img, **kwargs):
        # L
        L_feat = self.L_conv(img[:,:1,:,:])
        # AB
        AB_feat = self.AB_conv(img[:,1:,:,:])
        
        outputs = torch.cat((L_feat, AB_feat), dim=1)
        
        return outputs #(B, 3, H, W)
    
class InceptionCellA(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.layer1 = nn.Sequential(
            SimpleConv2d(input_dim, output_dim, kernel_size=1),
            SimpleConv2d(output_dim, output_dim, kernel_size=3, padding=1),
            SimpleConv2d(output_dim, output_dim, kernel_size=3, padding=1)
        )
        self.layer2 = nn.Sequential(
            SimpleConv2d(input_dim, output_dim, kernel_size=1),
            SimpleConv2d(output_dim, output_dim, kernel_size=3, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            SimpleConv2d(input_dim, output_dim, kernel_size=1)
        )
        self.layer4 = nn.Sequential(
            SimpleConv2d(input_dim, output_dim, kernel_size=1)
        )
        
    def forward(self, x):
        output = torch.cat((self.layer1(x), self.layer2(x), 
                   self.layer3(x), self.layer4(x)), dim=1)
        return output
    
class InceptionCellB(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.layer1 = nn.Sequential(
            SimpleConv2d(input_dim, output_dim, kernel_size=1),
            SimpleConv2d(output_dim, output_dim, kernel_size=(3,1), padding=(1,0)),
            SimpleConv2d(output_dim, output_dim, kernel_size=(1,3), padding=(0,1)),
            SimpleConv2d(output_dim, output_dim, kernel_size=(3,1), padding=(1,0)),
            SimpleConv2d(output_dim, output_dim, kernel_size=(1,3), padding=(0,1)),
        )
        self.layer2 = nn.Sequential(
            SimpleConv2d(input_dim, output_dim, kernel_size=1),
            SimpleConv2d(output_dim, output_dim, kernel_size=(3,1), padding=(1,0)),
            SimpleConv2d(output_dim, output_dim, kernel_size=(1,3), padding=(0,1)),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            SimpleConv2d(input_dim, output_dim, kernel_size=1)
        )
        self.layer4 = nn.Sequential(
            SimpleConv2d(input_dim, output_dim, kernel_size=1)
        )
        
    def forward(self, x):
        output = torch.cat((self.layer1(x), self.layer2(x), 
                   self.layer3(x), self.layer4(x)), dim=1)
        return output
    
class InceptionCellC(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.layer1 = nn.Sequential(
            SimpleConv2d(input_dim, output_dim, kernel_size=1),
            SimpleConv2d(output_dim, output_dim, kernel_size=3, padding=1),
            SimpleConv2d(output_dim, output_dim, kernel_size=(3,1), padding=(1,0)),
            SimpleConv2d(output_dim, output_dim, kernel_size=(1,3), padding=(0,1)),
        )
        self.layer2 = nn.Sequential(
            SimpleConv2d(input_dim, output_dim, kernel_size=1),
            SimpleConv2d(output_dim, output_dim, kernel_size=(3,1), padding=(1,0)),
            SimpleConv2d(output_dim, output_dim, kernel_size=(1,3), padding=(0,1)),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            SimpleConv2d(input_dim, output_dim, kernel_size=1)
        )
        self.layer4 = nn.Sequential(
            SimpleConv2d(input_dim, output_dim, kernel_size=1)
        )
        
    def forward(self, x):
        output = torch.cat((self.layer1(x), self.layer2(x), 
                   self.layer3(x), self.layer4(x)), dim=1)
        return output
    
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