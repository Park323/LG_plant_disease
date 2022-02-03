import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, argparse
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

from dataset import preprocess
from dataset.dataset import CustomDataset
from train import *
from utils.metric import *

import warnings
warnings.simplefilter('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_KOR = {idx:key for idx, key in enumerate(preprocess.label_description.values())}

def predict(config, model, dataset, training=False):
    model.eval()
    tqdm_dataset = tqdm(enumerate(dataset))
    results = []
    answer = []
    for batch, batch_item in tqdm_dataset:
        img = batch_item['img'].to(DEVICE)
        seq = batch_item['csv_feature'].to(DEVICE)
        with torch.no_grad():
            if 'decode' in dir(model):
                output = model.decode(img, seq, train=False)
            else:
                output = model(img, seq, train=False)
        results.extend(output)
        if training:
            answer.extend(batch_item['label'])
    results = torch.stack(results)
    return results, answer
    
def main(args):
    config = OmegaConf.load(f'config/config.yaml')
    mconfig = OmegaConf.load(f'config/{args.model_name}_config.yaml')
    
    TRAIN = mconfig.TRAIN
    TEST  = config.TEST
    DATA  = config.DATA
    
    IMAGE_PATH = DATA.DATA_ROOT + '/' + DATA.IMAGE_PATH
    TRAIN_PATH = DATA.DATA_ROOT + '/' + DATA.TRAIN_PATH
    VALID_PATH = DATA.DATA_ROOT + '/' + DATA.VALID_PATH
    TEST_PATH  = DATA.DATA_ROOT + '/' + DATA.TEST_PATH
    
    ##########################      DataLoader 정의     #########################
    print('Data Loading...')
    preprocessor = get_preprocessor(TRAIN, args.model_name)
    with open(TEST_PATH, 'r') as f:
        test_dataset = CustomDataset(f.read().split('\n'), pre=preprocessor, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=TEST.BATCH_SIZE, num_workers=TEST.NUM_WORKER, shuffle=False)
    
    ##################            Define metrics          #######################
    
    criterion, metric_function = get_metrics(args.model_name, smoothing=args.smoothing, gamma=args.gamma)
    
    ##################              Inference             #######################
    assert args.model_path != 'none', 'Model Path should be passed by argument.\nExample) train.py -i -ip output/sample_model.pt'
    print('Inference Start...')
    
    model = torch.load(args.model_path)
    
    preds, answer = predict(TEST, model, test_dataloader)
    
    submission = pd.read_csv(f'{TEST.SAMPLE_PATH}')
    submission['label'] = metric_function(None, preds, inference=True, preprocess=preprocessor)
    submission.to_csv(f'{TRAIN.SAVE_PATH}/submission.csv', index=False)
    
    return None
            
if __name__=='__main__':
    
    # Load Config
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-ip','--model_path', default='none')
    
    args = parser.parse_args()
    
    main(args)