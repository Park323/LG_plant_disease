import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from glob import glob
import os, json, pickle, argparse
from omegaconf import OmegaConf

import torch
from torch import nn
from torchvision import models
from sklearn.metrics import f1_score

from dataset import preprocess
from dataset.dataset import CustomDataset
from model.base_model import CNN2RNN
from metric.metric import accuracy_function

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
    
    # Load Config
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load_model',
                        type=str, required=True)
    args = parser.parse_args()
    
    config = OmegaConf.load('config/config.yaml')
    TEST = config.TEST
    DATA = config.DATA
    
    preprocessor = preprocess.Base_Processer(config)
    preprocessor.load_dictionary(f'{DATA.DATA_ROOT}/prepro_dict.pkl')
    
    with open(f"{DATA.DATA_ROOT}/{DATA.TEST_PATH}", 'r') as f:
        test_dataset = CustomDataset(f.read().split('\n'), pre=preprocessor)
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST.BATCH_SIZE, num_workers=TEST.NUM_WORKER, shuffle=False)
    
    model = torch.load(args.load_model)
    
    preds, answer = predict(TEST, model, test_dataloader)
    
    score = accuracy_function(answer, preds)
    print(f"f1-score : {score}")