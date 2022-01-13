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
from utils.metric import *

import warnings
warnings.simplefilter('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(config, model, dataset, training=False):
    model.eval()
    tqdm_dataset = tqdm(enumerate(dataset))
    training = False
    results = []
    answer = []
    for batch, batch_item in tqdm_dataset:
        img = batch_item['img'].to(DEVICE)
        seq = batch_item['csv_feature'].to(DEVICE)
        with torch.no_grad():
            output = model(img, seq, train=False)
        results.extend(output)
        if training:
            answer.extend(batch_item['label'])
    return results, answer

if __name__=='__main__':
    
    # Load Config
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load_model',
                        type=str, required=True)
    parser.add_argument('-m', '--model_name', default='base',
                        type=str, required=True)
    parser.add_argument('-t', '--training', action='store_true')
    args = parser.parse_args()
    
    config = OmegaConf.load('config/config.yaml')
    TEST = config.TEST
    DATA = config.DATA
    training = args.training
    
    if args.model_name=='base':
        preprocessor = preprocess.Base_Processor(config)
        criterion = base_loss
        metric_function = accuracy_function
    elif args.model_name=='lab':
        preprocessor = preprocess.LAB_Processor(config)
        criterion = lab_loss
        metric_function = lab_metric
    elif args.model_name=='drj':
        preprocessor = preprocess.Base_Processor2(config)
        criterion = seperated_loss
        metric_function = seperated_metric
        
    if os.path.exists(f'{DATA.DATA_ROOT}/{preprocessor.dict_name}'):
        preprocessor.load_dictionary(f'{DATA.DATA_ROOT}/{preprocessor.dict_name}')
    else:
        preprocessor.initialize()
    
    with open(f"{DATA.DATA_ROOT}/{DATA.TEST_PATH}", 'r') as f:
        test_dataset = CustomDataset(f.read().split('\n'), pre=preprocessor, mode='test')
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST.BATCH_SIZE, num_workers=TEST.NUM_WORKER, shuffle=False)
    
    model = torch.load(args.load_model)
    
    preds, answer = predict(TEST, model, test_dataloader, training)
    
    if training:
        score = metric_function(answer, preds)
        print(f"f1-score : {score}")
    
    submission = pd.read_csv(f'{TEST.SAMPLE_PATH}')
    submission['label'] = metric_function(None, preds, preprocessor, inference=True)
    submission.to_csv(f'{TEST.SUBMIT_PATH}', index=False)