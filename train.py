import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from glob import glob
import os, argparse
from omegaconf import OmegaConf

import torch
from torch import nn

from dataset import preprocess
from dataset.dataset import CustomDataset, get_annotations
from model.base_model import CNN2RNN
from model.jk_model import DrJeonko
from model.lab_model import CatClassifier, LAB_model
from utils.scheduler import CosineAnnealingWarmUpRestarts
from utils.metric import *

import warnings
warnings.simplefilter('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_step(model, criterion, optimizer, batch_item, training, preprocess=None, metric_function=accuracy_function):
    img = batch_item['img'].to(DEVICE)
    csv_feature = batch_item['csv_feature'].to(DEVICE)
    label = batch_item['label']
    label = label.to(DEVICE) if isinstance(label, torch.Tensor) else [item.to(DEVICE) for item in label]
    
    if training is True:
        # annotations = [get_annotations(path, preprocess.json_processing) for path in batch_item['json_path']]
        
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(img, csv_feature, labels=label)
            loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        score = metric_function(label, output, preprocess)
        return loss, score
    else:
        model.eval()
        with torch.no_grad():
            output = model(img, csv_feature, labels=label, train=False)
            loss = criterion(output, label)
        score = metric_function(label, output, preprocess)
        return loss, score

if __name__=='__main__':
    
    # Load Config
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--from_epoch',
                        type=int, default=0)
    parser.add_argument('-m', '--model_name',
                        type=str, default='base')
    parser.add_argument('-s','--for_submission', action='store_true')
    
    args = parser.parse_args()
    config = OmegaConf.load(f'config/config.yaml')
    mconfig = OmegaConf.load(f'config/{args.model_name}_config.yaml')
    
    TRAIN = mconfig.TRAIN
    DATA  = config.DATA
    
    IMAGE_PATH = DATA.DATA_ROOT + '/' + DATA.IMAGE_PATH
    TRAIN_PATH = DATA.DATA_ROOT + '/' + DATA.TRAIN_PATH
    VALID_PATH = DATA.DATA_ROOT + '/' + DATA.VALID_PATH
    
    ##########################      DataLoader 정의     #########################
    print('Data Loading...')
    
    if args.model_name=='base':
        preprocessor = preprocess.Base_Processor(config)
    elif args.model_name=='lab':
        preprocessor = preprocess.LAB_Processor(config)
    elif args.model_name=='lab_cat':
        preprocessor = preprocess.LAB_Processor(config)
    elif args.model_name=='drj':
        preprocessor = preprocess.Base_Processor2(config)
        
    if os.path.exists(f'{DATA.DATA_ROOT}/{preprocessor.dict_name}'):
        preprocessor.load_dictionary(f'{DATA.DATA_ROOT}/{preprocessor.dict_name}')
    else:
        preprocessor.initialize()
    
    if args.for_submission:
        with open(TRAIN_PATH, 'r') as f:
            train_paths = f.read().split('\n')
        with open(VALID_PATH, 'r') as f:
            train_paths = [*train_paths, *f.read().split('\n')]
        train_dataset = CustomDataset(train_paths, pre=preprocessor)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN.BATCH_SIZE, 
                                                    num_workers=config.TRAIN.NUM_WORKER, shuffle=True)
    else:
        with open(TRAIN_PATH, 'r') as f:
            train_dataset = CustomDataset(f.read().split('\n'), pre=preprocessor)
        with open(VALID_PATH, 'r') as f:
            val_dataset = CustomDataset(f.read().split('\n'), pre=preprocessor)
            
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN.BATCH_SIZE, 
                                                    num_workers=config.TRAIN.NUM_WORKER, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=TRAIN.BATCH_SIZE, 
                                                    num_workers=config.TRAIN.NUM_WORKER, shuffle=False)
    #####################################################################
    
    ################  Model / Loss / Optimizer / Scheduler 정의  ################
    print('Model Loading...')
    
    if args.from_epoch:
        assert os.path.exists(TRAIN.SAVE_PATH + '/' + f'model_{args.from_epoch}.pt'), f'Model is not Exists: {TRAIN.SAVE_PATH}/model_{args.from_epoch}.pt'
        model = torch.load(TRAIN.SAVE_PATH + '/' + f'model_{args.from_epoch}.pt')
    else:
        # Argument에 따라 model 변경
        if args.model_name=='base':
            model = CNN2RNN(max_len=config.TRAIN.MAX_LEN, embedding_dim=TRAIN.EMBEDDING_DIM, \
                            num_features=TRAIN.NUM_FEATURES, class_n=TRAIN.CLASS_N, \
                            rate=TRAIN.DROPOUT_RATE)
        elif args.model_name=='lab':
            model = LAB_model(TRAIN)
        elif args.model_name=='lab_cat':
            model = CatClassifier(TRAIN)
        elif args.model_name=='drj':
            model = DrJeonko(TRAIN)

    model = model.to(DEVICE)
    
    if args.model_name=='base':
        criterion = base_loss
        metric_function = accuracy_function
    elif args.model_name=='lab':
        criterion = lab_dr_loss
        metric_function = lab_metric
    elif args.model_name=='lab_cat':
        criterion = lab_cat_loss
        metric_function = lab_cat_metric
    elif args.model_name=='drj':
        criterion = seperated_loss
        metric_function = seperated_metric
    
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN.LR_MIN)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=15, T_mult=2, 
                                              eta_max=TRAIN.LR_MAX, T_up=3, gamma=0.5)
    
    if args.from_epoch and os.path.exists(f'{TRAIN.SAVE_PATH}/optimizer_states.pt'):
        optimizer.load_state_dict(torch.load(f'{TRAIN.SAVE_PATH}/optimizer_states.pt'))
        scheduler.load_state_dict(torch.load(f'{TRAIN.SAVE_PATH}/scheduler_states.pt'))
    #############################################################################
    
    
    #########################         TRAIN         #############################
    print('Train Start')
    
    if args.from_epoch and os.path.exists(f'{TRAIN.SAVE_PATH}/train_history.pt'):
        hists = torch.load(f'{TRAIN.SAVE_PATH}/train_history.pt')
        loss_plot, val_loss_plot, metric_plot, val_metric_plot = hists.values()
    else:
        loss_plot, val_loss_plot = [], []
        metric_plot, val_metric_plot = [], []

    epoch_from = args.from_epoch

    for epoch in range(epoch_from, TRAIN.EPOCHS):
        total_loss, total_val_loss = 0, 0
        total_acc, total_val_acc = 0, 0
        
        tqdm_dataset = tqdm(enumerate(train_dataloader))
        training = True
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_acc = train_step(model, criterion, optimizer, batch_item, training, 
                                               preprocess=preprocessor, metric_function=metric_function)
            total_loss += batch_loss
            total_acc += batch_acc
            
            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Loss': '{:06f}'.format(batch_loss.item()),
                'Mean Loss' : '{:06f}'.format(total_loss/(batch+1)),
                'Mean F-1' : '{:06f}'.format(total_acc/(batch+1)),
                'Learning_rate' : '{}'.format(optimizer.param_groups[0]['lr'])
            })
        
        if args.for_submission:
            total_val_loss = 0
            total_val_acc = 0
        else:
            tqdm_dataset = tqdm(enumerate(val_dataloader))
            training = False
            for batch, batch_item in tqdm_dataset:
                batch_loss, batch_acc = train_step(model, criterion, optimizer, batch_item, training, 
                                                   preprocess=preprocessor, metric_function=metric_function)
                total_val_loss += batch_loss
                total_val_acc += batch_acc
                
                tqdm_dataset.set_postfix({
                    'Epoch': epoch + 1,
                    'Val Loss': '{:06f}'.format(batch_loss.item()),
                    'Mean Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                    'Mean Val F-1' : '{:06f}'.format(total_val_acc/(batch+1))
                })
        
        scheduler.step(epoch)
        
        if len(loss_plot)==epoch:
            loss_plot.append(total_loss.item()/(batch+1))
            metric_plot.append(total_acc/(batch+1))
            val_loss_plot.append(total_val_loss.item()/(batch+1))
            val_metric_plot.append(total_val_acc/(batch+1))
        else:
            loss_plot[epoch]=total_loss.item()/(batch+1)
            metric_plot[epoch]=total_acc/(batch+1)
            val_loss_plot[epoch]=total_val_loss.item()/(batch+1)
            val_metric_plot[epoch]=total_val_acc/(batch+1)
                
        # check/make directory for model file
        dp = []
        for directory in TRAIN.SAVE_PATH.split('/'):
            if directory:
                new_dir = '/'.join([*dp, directory])
                if not os.path.exists(new_dir):
                    os.mkdir(new_dir)
            dp.append(directory)
        
        
        if total_val_loss > max(val_metric_plot):
            torch.save(model, f'{TRAIN.SAVE_PATH}/model_best.pt')
        
        if ((epoch+1) % config.TRAIN.SAVE_PERIOD == 0) or (epoch+1 == TRAIN.EPOCHS):
            torch.save(model, f'{TRAIN.SAVE_PATH}/model_{epoch+1}.pt')
            torch.save({'train_loss':loss_plot, 'val_loss':val_loss_plot,
                        'train_f1':metric_plot, 'val_f1':val_metric_plot},
                        f'{TRAIN.SAVE_PATH}/train_history.pt')
            torch.save(scheduler.state_dict(),
                        f'{TRAIN.SAVE_PATH}/scheduler_states.pt')
            torch.save(optimizer.state_dict(),
                        f'{TRAIN.SAVE_PATH}/optimizer_states.pt')