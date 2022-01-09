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

from dataset.dataset import CustomDataset
from model.base_model import CNN2RNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 제공된 sample data는 파프리카와 시설포도 2종류의 작물만 존재
label_description = {
'3_00_0': '파프리카_정상',
'3_a9_1': '파프리카흰가루병_초기',
'3_a9_2': '파프리카흰가루병_중기',
'3_a9_3': '파프리카흰가루병_말기',
'3_a10_1': '파프리카잘록병_초기',
'3_a10_2': '파프리카잘록병_중기',
'3_a10_3': '파프리카잘록병_말기',
'3_b3_1': '칼슘결핍_초기',
'3_b3_2': '칼슘결핍_중기',
'3_b3_3': '칼슘결핍_말기',
'3_b6_1': '다량원소결핍 (N)_초기',
'3_b6_2': '다량원소결핍 (N)_중기',
'3_b6_3': '다량원소결핍 (N)_말기',
'3_b7_1': '다량원소결핍 (P)_초기',
'3_b7_2': '다량원소결핍 (P)_중기',
'3_b7_3': '다량원소결핍 (P)_말기',
'3_b8_1': '다량원소결핍 (K)_초기',
'3_b8_2': '다량원소결핍 (K)_중기',
'3_b8_3': '다량원소결핍 (K)_말기',
'6_00_0': '시설포도_정상',
'6_a11_1': '시설포도탄저병_초기',
'6_a11_2': '시설포도탄저병_중기',
'6_a11_3': '시설포도탄저병_말기',
'6_a12_1': '시설포도노균병_초기',
'6_a12_2': '시설포도노균병_중기',
'6_a12_3': '시설포도노균병_말기',
'6_b4_1': '일소피해_초기',
'6_b4_2': '일소피해_중기',
'6_b4_3': '일소피해_말기',
'6_b5_1': '축과병_초기',
'6_b5_2': '축과병_중기',
'6_b5_3': '축과병_말기',
}

def accuracy_function(real, pred):    
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score

def train_step(model, criterion, optimizer, batch_item, training):
    img = batch_item['img'].to(DEVICE)
    csv_feature = batch_item['csv_feature'].to(DEVICE)
    label = batch_item['label'].to(DEVICE)
    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(img, csv_feature)
            loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        score = accuracy_function(label, output)
        return loss, score
    else:
        model.eval()
        with torch.no_grad():
            output = model(img, csv_feature)
            loss = criterion(output, label)
        score = accuracy_function(label, output)
        return loss, score

if __name__=='__main__':
    
    # Load Config
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--from_epoch',
                        type=int, default=0)
    
    args = parser.parse_args()
    config = OmegaConf.load('config/config.yaml')
    
    TRAIN = config.TRAIN
    DATA  = config.DATA
    
    TRAIN_PATH = DATA.DATA_PATH + '/' + config.DATA.TRAIN_PATH
    VALID_PATH = DATA.DATA_PATH + '/' + config.DATA.VALID_PATH
    
    # 분석에 사용할 feature 선택
    csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고', 
                    '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']

    csv_files = sorted(glob(DATA.DATA_PATH + '/*/*.csv'))

    temp_csv = pd.read_csv(csv_files[0])[csv_features]
    max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

    # feature 별 최대값, 최솟값 계산
    for csv in tqdm(csv_files[1:]):
        temp_csv = pd.read_csv(csv)[csv_features]
        temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
        max_arr = np.max([max_arr,temp_max], axis=0)
        min_arr = np.min([min_arr,temp_min], axis=0)

    # feature 별 최대값, 최솟값 dictionary 생성
    csv_feature_dict = {csv_features[i]:[min_arr[i], max_arr[i]] for i in range(len(csv_features))}
    
    label_encoder = {key:idx for idx, key in enumerate(label_description)}
    label_decoder = {val:key for key, val in label_encoder.items()}
    
    with open(DATA.DATA_PATH+'/'+'feats_minmax_and_label_dict.pkl', 'wb') as f:
        pickle.dump({'csv_feature_dict':csv_feature_dict, 
                     'label_encoder':label_encoder}, f)
    
    with open(DATA.DATA_PATH+'/'+config.DATA.TRAIN_PATH, 'r') as f:
        train_dataset = CustomDataset(f.read().split(','), label_encoder=label_encoder, csv_feature_dict=csv_feature_dict)
    with open(DATA.DATA_PATH+'/'+config.DATA.VALID_PATH, 'r') as f:
        val_dataset = CustomDataset(f.read().split(','), label_encoder=label_encoder, csv_feature_dict=csv_feature_dict)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN.BATCH_SIZE, num_workers=16, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=TRAIN.BATCH_SIZE, num_workers=16, shuffle=False)
    
    if args.from_epoch:
        if os.path.exists(TRAIN.SAVE_PATH + '/' + f'model_{args.epoch_from}.pt'):
            model = torch.load(TRAIN.SAVE_PATH + '/' + f'model_{args.epoch_from}.pt')
        else:
            assert f'Model is not Exists: {TRAIN.SAVE_PATH}/model_{args.epoch_from}.pt'
    else:
        model = CNN2RNN(max_len=TRAIN.MAX_LEN, embedding_dim=TRAIN.EMBEDDING_DIM, \
                        num_features=TRAIN.NUM_FEATURES, class_n=TRAIN.CLASS_N, \
                        rate=TRAIN.DROPOUT_RATE)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    loss_plot, val_loss_plot = [], []
    metric_plot, val_metric_plot = [], []

    epoch_from = args.from_epoch

    for epoch in range(epoch_from, TRAIN.EPOCHS):
        total_loss, total_val_loss = 0, 0
        total_acc, total_val_acc = 0, 0
        
        tqdm_dataset = tqdm(enumerate(train_dataloader))
        training = True
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_acc = train_step(model, criterion, optimizer, batch_item, training)
            total_loss += batch_loss
            total_acc += batch_acc
            
            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Loss': '{:06f}'.format(batch_loss.item()),
                'Mean Loss' : '{:06f}'.format(total_loss/(batch+1)),
                'Mean F-1' : '{:06f}'.format(total_acc/(batch+1))
            })
        loss_plot.append(total_loss/(batch+1))
        metric_plot.append(total_acc/(batch+1))
        
        tqdm_dataset = tqdm(enumerate(val_dataloader))
        training = False
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_acc = train_step(model, criterion, optimizer, batch_item, training)
            total_val_loss += batch_loss
            total_val_acc += batch_acc
            
            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Val Loss': '{:06f}'.format(batch_loss.item()),
                'Mean Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                'Mean Val F-1' : '{:06f}'.format(total_val_acc/(batch+1))
            })
        val_loss_plot.append(total_val_loss/(batch+1))
        val_metric_plot.append(total_val_acc/(batch+1))
                
        # check/make directory for model file
        dp = []
        for directory in config.SAVE_PATH.split('/'):
            new_dir = '/'.join([*dp, directory])
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            dp.append(directory)
        
        # if np.max(val_metric_plot) == val_metric_plot[-1]:
        torch.save(model, TRAIN.SAVE_PATH + '/' + f'model_{epoch+1}.pt')
        
