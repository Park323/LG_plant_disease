import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.transforms import Resize
from torchvision import models

class DrJeonko(nn.Module):
    def __init__(self, config):
        assert config.EMBEDDING_DIM % 2 == 0, 'EMBEDDING_DIM should be even number!!!'
        super(DrJeonko, self).__init__()
        self.cnn = CropClassifier(config)
        self.rnn = EnvAnalyzer(config)
        self.detector = BlightDetector(config)
        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)
        self.fc_crop = nn.Sequential(nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM),
                                     nn.ReLU(),
                                     nn.Dropout(p=config.DROPOUT_RATE),
                                     nn.Linear(config.EMBEDDING_DIM, config.CROP_N),
                                     nn.ReLU())
        self.fc_disease = nn.Sequential(nn.Linear(config.EMBEDDING_DIM * 3, config.EMBEDDING_DIM),
                                        nn.ReLU(),
                                        nn.Dropout(p=config.DROPOUT_RATE),
                                        nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM),
                                        nn.ReLU(),
                                        nn.Dropout(p=config.DROPOUT_RATE),
                                        nn.Linear(config.EMBEDDING_DIM, config.CLASS_N),
                                        nn.ReLU())
        self.fc_risk = nn.Sequential(nn.Linear(config.EMBEDDING_DIM, int(config.EMBEDDING_DIM/2)),
                                     nn.ReLU(),
                                     nn.Dropout(p=config.DROPOUT_RATE),
                                     nn.Linear(int(config.EMBEDDING_DIM/2), 1),
                                     nn.ReLU())
        
    def forward(self, img, seq, json=None, train=True, **kwargs):
        feature1 = self.cnn(img)
        output1 = self.fc_crop(feature1)
        feature2 = self.rnn(feature1, seq)
        feature3 = self.detector(img, json, train=train)
        output2 = self.fc_risk(feature3)
        output3 = self.fc_disease(torch.cat([feature1, feature2, feature3], dim=0))
        
        return (output1, output2, output3)
    
class CropClassifier(nn.Module):
    def __init__(self, config):
        super(CropClassifier, self).__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, config.FEATURE_DIM)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)
    
    def forward(self, img):
        features = self.inception(img)

        for name, param in self.inception.named_parameters():
            if 'fc.weight' in name or 'fc.bias' in name:
                param.requires_grad = True
            else :
                param.requires_grad = False

        return self.dropout(self.relu(features))
    
class EnvAnalyzer(nn.Module):
    def __init__(self, config):
        super(EnvAnalyzer, self).__init__()
        self.in_fc  = nn.Linear(config.FEATURE_DIM, config.EMBEDDING_DIM * config.N_LAYER)
        self.rnn    = nn.GRU(input_size=config.SEQ_FEATURE_N, hidden_size=int(config.EMBEDDING_DIM/2), 
                           num_layers= config.N_LAYER, batch_first=True, 
                           dropout=config.DROPOUT_RATE, bidirectional=True)
        self.out_fc = nn.Linear(config.EMBEDDING_DIM * config.N_LAYER, config.EMBEDDING_DIM)
        self.relu   = nn.ReLU()
        self.dropout= nn.Dropout(p=config.DROPOUT_RATE)
    
    def forward(self, feat, seq):
        feat   = self.dropout(self.relu(self.in_fc(feat)))
        feat   = feat.view(feat.shape[0], self.rnn.num_layers * 2, self.rnn.hidden_size)
        feat   = feat.transpose(0,1)
        _, output = self.rnn(seq, feat)
        output = output.transpose(0,1)
        output = output.view(output.shape[0], -1)
        output = self.dropout(self.relu(self.out_fc(output)))
        
        return output

class BlightDetector(nn.Module):
    def __init__(self, config):
        super(BlightDetector, self).__init__()
        self.batchsize= config.BATCH_SIZE
        self.detector = None
        self.yolo     = config.YOLO_PATH
        self.cnn1     = nn.Conv2d(3, 16, 3, padding=1)
        self.cnn2     = nn.Conv2d(16, 64, 3, padding=1)
        self.cnn3     = nn.Conv2d(64, 1, 3, padding=1)
        self.rnn      = nn.GRU(input_size=config.BLIGHT_DIM, hidden_size=config.EMBEDDING_DIM, 
                                num_layers=1, batch_first=True, 
                                dropout=config.DROPOUT_RATE, bidirectional=False)
        self.initial  = torch.zeros((1, config.BATCH_SIZE, config.EMBEDDING_DIM), 
                                    dtype=torch.float32, requires_grad=True)
        self.fc       = nn.Linear(1*32*32, config.BLIGHT_DIM)
        self.dropout  = nn.Dropout(p=config.DROPOUT_RATE)
        self.relu     = nn.ReLU()
        
    def forward(self, img, annots, train=True):
        if not train:
            if self.detector:
                pass
            else:
                self.detector = torch.load(self.yolo)
            annots = self.detector(img)
        print(annots)
        print(type(annots))
        
        #Image Crop
        blights = torch.zeros((img.shape[0], 3, 32, 32))
        for i, aimg, annot in enumerate(zip(img, annots)):
            annot = annot.split()
            part = aimg[:,annot[1]:annot[1]+annot[3], annot[0]:annot[0]+annot[2]].detach()
            part = Resize((32,32))(part)
            blights[i] = part
        
        blight_features = []
        for batch in enumerate(blights):
            blight_features.append(self.cnn3(self.cnn2(self.cnn1(batch))))
        blight_features = torch.cat(blight_features, dim=0)
        
        blight_features = blight_features.view(-1, 1*32*32)
        
        blight_features = self.fc(blight_features)
        
        _, feature = self.rnn(blight_features.view(-1,self.rnn.hidden_size), self.initial)
        feature = feature.transpose(0,1)
        
        feature = self.dropout(self.relu(feature.squeeze(1)))
        return feature