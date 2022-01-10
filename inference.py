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

from dataset import preprocess
from dataset.dataset import CustomDataset
from model.base_model import CNN2RNN

import warnings
warnings.simplefilter('ignore')

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
    
    preprocessor = preprocess.Base_Processer(config)
    preprocessor.load_dictionary(f'{config.DATA.DATA_ROOT}/prepro_dict.pkl')
    
    with open(f"{DATA.DATA_ROOT}/{DATA.TEST_PATH}", 'r') as f:
        test_dataset = CustomDataset(f.read().split('\n'), pre=preprocessor)
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=TRAIN.BATCH_SIZE, num_workers=TRAIN.NUM_WORKER, shuffle=False)
    
    model = torch.load(TRAIN.SAVE_PATH)
    
    preds, answer = predict(TRAIN, model, test_dataloader)
    
    print(f"{preds} : {answer}")