import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import transforms
import numpy as np

class LAB_model(nn.Module):
    def __init__(self, config):
        super(LAB_model,self).__init__()
        self.classifier = torch.load(f"{config.ClASSIFIER}")
        for param in self.classifier.parameters():
            param.requires_grad = False
        # self.classifier = CatClassifier(config)
        self.labfeature = LabExtractor(config)
        self.conv = nn.Sequential(
            SimpleConv2d(67,80, kernel_size=3, padding=1, bias=False),
            SimpleConv2d(80,192, kernel_size=3, bias=False),
            nn.MaxPool2d(3,2)
        )
        self.Inceptionx3A = nn.Sequential(
            InceptionCellA(192, 72),
            InceptionCellA(72*4, 72),
            InceptionCellA(72*4, 72),
            nn.MaxPool2d(3, 2)
        )
        self.Inceptionx3B = nn.Sequential(
            InceptionCellB(72*4, 192),
            InceptionCellB(192*4, 192),
            InceptionCellB(192*4, 192)
        )
        self.crop_map = nn.Linear(6, 54*54)
        self.area_map = nn.Linear(7, 54*54)
        self.disease_map = nn.Linear(768, 21)
        self.risk_map = nn.Linear(768, 4)
        
    def forward(self, img, seq, labels=None, train=True, **kwargs):
        if train:
            features = torch.cat((labels[:6],labels[31:]),dim=1).detach()
        else:
            features = self.classifier(img).detach() 
        crop, area = features[:,:6], features[:,6:13]
        c_map = self.crop_map(crop).view(-1,1,54,54)
        a_map = self.area_map(area).view(-1,1,54,54)
        
        LAB = self.labfeature(img)
        
        inp = torch.cat((LAB, c_map, a_map), dim=1) #(BATCH_SIZE, 66, 54, 54)
        
        inp = self.conv(inp)
        inp = self.Inceptionx3A(inp)
        inp = self.Inceptionx3B(inp)
        
        inp = F.adaptive_avg_pool2d(inp, (1,1))
        out = inp.view(inp.shape[0], -1)
        
        out_d = F.softmax(self.disease_map(out))
        out_r = F.softmax(self.risk_map(out))
        
        outputs = torch.cat((crop, out_d, out_r, area), dim=1)
        
        return outputs
    
class CatClassifier(nn.Module):
    def __init__(self, config):
        super(CatClassifier,self).__init__()
        # self.densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        # for param in self.densenet.parameters():
        #     param.requires_grad = False
        # self.conv     = nn.Sequential(
        #     SimpleConv2d(1024, 512, kernel_size=3, padding=1, bias=False),
        #     SimpleConv2d(512, 128, kernel_size=3, padding=1, bias=False),
        #     SimpleConv2d(128, 22, kernel_size=3, padding=1, bias=False),
        # )
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, 512)
        for name, param in self.inception.named_parameters():
            if 'fc.weight' in name or 'fc.bias' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.crop_fc  = nn.Linear(512, 6)
        self.area_fc  = nn.Linear(512, 7)
        
    def forward(self, img, seq, labels=None, train=True, **kwargs):
        # # densenet
        # features = self.densenet.features(img)
        # out = F.relu(features, inplace=True)
        
        # out = self.conv(out)
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        # out = out.view(out.shape[0], -1)
        # out_c = F.softmax(out[:,:6])
        # out_a = F.softmax(out[:,6:6+7])
        
        img = transforms.Resize(256)(img)
        feat = self.inception(img)
        out_c = F.softmax(self.crop_fc(feat))
        out_a = F.softmax(self.area_fc(feat))
        
        outputs = torch.cat((out_c, out_a), dim=1)
        
        return outputs
    
class LabExtractor(nn.Module):
    def __init__(self, config):
        super(LabExtractor,self).__init__()
        self.L_conv = nn.Sequential(
            SimpleConv2d(1,6, kernel_size=3, stride=2, bias=False),
            SimpleConv2d(6,6, kernel_size=3, bias=False),
            SimpleConv2d(6,13, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(2,2)
        )
        self.AB_conv = nn.Sequential(
            SimpleConv2d(2,26, kernel_size=3, stride=2, bias=False),
            SimpleConv2d(26,26, kernel_size=3, bias=False),
            SimpleConv2d(26,51, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(2,2)
        )
        
    def forward(self, img, seq, labels=None, train=True, **kwargs):
        # Save device
        device = img.device
        
        # LAB
        img = xyz2lab(rgb2xyz(img.permute(0,2,3,1).cpu().detach().numpy()))
        img = torch.from_numpy(img).to(torch.float32).permute(0,3,1,2).contiguous()
        img = img.to(device)

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
    
def rgb2xyz(img):
    # https://www.easyrgb.com/en/math.php
    # input is numpy (B, W, H, C)
    # sR, sG and sB (Standard RGB) input range = 0 ÷ 255
    # X, Y and Z output refer to a D65/2° standard illuminant.
    
    mask = img > 0.04045
    img[mask] = np.power((img[mask] + 0.055) / 1.055, 2.4)
    img[~mask] /= 12.92
    
    img *= 100
    
    xyz_conv = np.array([[0.4124, 0.3576, 0.1805],
                         [0.2126, 0.7152, 0.0722],
                         [0.0193, 0.1192, 0.9505]])
    
    return img @ xyz_conv.T

def xyz2lab(img):
    # https://www.easyrgb.com/en/math.php
    # input is tensor (B, C, W, H)
    # Reference-X, Y and Z refer to specific illuminants and observers.
    # Common reference values are available below in this same page.
    refX, refY, refZ = 0.95047, 1., 1.08883   # This was: `lab_ref_white` D65 / 2
    
    img[:,:,:,0] /= refX
    img[:,:,:,1] /= refY
    img[:,:,:,2] /= refZ

    mask = img > 0.008856
    
    img[mask] = np.power(img[mask], 1/3)
    img[~mask] = 7.787 * img[~mask] + 16/116

    lab_conv = np.array([[0,    116,    0],
                         [500, -500,    0],
                         [0,    200, -200]])

    img = img @ lab_conv.T + np.array([-16, 0, 0])
    
    # import matplotlib.pyplot as plt
    # plt.imshow(img[0,:,:,0], cmap='gray')
    # plt.show()
    # plt.imshow(img[0,:,:,1], cmap='gray')
    # plt.show()
    # plt.imshow(img[0,:,:,2], cmap='gray')
    # plt.show()
    
    return img