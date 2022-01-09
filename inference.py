import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from glob import glob
import os, json, pickle
from omegaconf import OmegaConf

import torch
from torch import nn
from torchvision import models
from sklearn.metrics import f1_score

from dataset.dataset import CustomDataset
from model.base_model import CNN2RNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(config, model, dataset):
    model.eval()
    tqdm_dataset = tqdm(enumerate(dataset))
    training = False
    results = []
    answer = []
    for batch, batch_item in tqdm_dataset:
        img = batch_item['img'].to(DEVICE)
        seq = batch_item['csv_feature'].to(DEVICE)
        with torch.no_grad():
            output = model(img, seq)
        output = torch.tensor(torch.argmax(output, axis=-1), dtype=torch.int32).cpu().numpy()
        results.extend(output)
        answer.extend(batch_item['label'])
    return results, answer

if __name__=='__main__':
    
    config = OmegaConf.load('config/config.yaml')
    TRAIN = config.TRAIN
    DATA = config.DATA
    
    with open(DATA.DATA_PATH+'/'+'feats_minmax_and_label_dict.pkl', 'rb') as f:
        param_set = pickle.load(f)
    csv_feature_dict, label_encoder = param_set.values()
    
    with open(DATA.DATA_PATH+'/'+DATA.TEST_PATH, 'r') as f:
        test_dataset = CustomDataset(f.read().split(','), label_encoder=label_encoder, csv_feature_dict=csv_feature_dict)
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=TRAIN.BATCH_SIZE, num_workers=16, shuffle=False)
        
    model = torch.load(TRAIN.SAVE_PATH)
    
    preds, answer = predict(TRAIN, model, test_dataloader)