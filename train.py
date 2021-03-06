import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os, argparse
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

from dataset import preprocess
from dataset.dataset import CustomDataset
from model.base_model import CNN2RNN
from model.jk_model import DrJeonko
from model.lab_model import *
from model.dense_model import DenseNet
from model.vit_model import MyViT, ViT_tuned, ImToSeqTransformer
from utils.scheduler import CosineAnnealingWarmUpRestarts
from utils.metric import *

import warnings
warnings.simplefilter('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_KOR = {idx:key for idx, key in enumerate(preprocess.label_description.values())}

def train_step(
    model, criterion, optimizer, 
    batch_item, preprocess=None, metric_function=accuracy_function, num_class=None, **kwargs):
    
    img = batch_item['img'].to(DEVICE)
    csv_feature = batch_item['csv_feature'].to(DEVICE)
    label = batch_item['label']
    label = label.to(DEVICE)
    
    model.train()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        output = model(img, csv_feature, labels=label)
        loss = criterion(output, label, **kwargs)
        
    if math.isnan(loss):
        raise f'Loss has NaN value with output\n{output}'
    
    loss.backward()
    optimizer.step()
    if 'decode' in dir(model):
        model.eval()
        with torch.no_grad():
            output = model.decode(img, csv_feature)
    score, correct, total = metric_function(label, output, preprocess=preprocess, num_class=num_class)
    return loss, score, correct, total

def valid_step(
    model, criterion, optimizer, 
    batch_item, preprocess=None, metric_function=accuracy_function, num_class=None, **kwargs):
    assert num_class, 'Argument num_class is needed!!'
    
    img = batch_item['img'].to(DEVICE)
    csv_feature = batch_item['csv_feature'].to(DEVICE)
    label = batch_item['label']
    label = label.to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        output = model(img, csv_feature, labels=label, train=False)
        loss = criterion(output, label, **kwargs)
    if 'decode' in dir(model):
        output = model.decode(img, csv_feature)
    score, correct, total = metric_function(label, output, preprocess=preprocess, num_class=num_class)
    return loss, score, correct, total

def get_preprocessor(config, model_name=None, **kwargs):
    if model_name=='base':
        return preprocess.Basic_CSV_Processor(config)
    elif model_name=='lab':
        return preprocess.LAB_Processor(config)
    elif model_name=='dense':
        return preprocess.Dense_Processor(config)
    elif model_name=='vit':
        return preprocess.ViT_Processor(config)
    elif model_name=='imseq':
        return preprocess.Seq_Processor(config)

def get_metrics(model_name, **_kwargs):
    
    criterion = lambda *args, **kwargs: ce_loss(*args, **kwargs, **_kwargs)
    metric_function = lambda *args, **kwargs: accuracy_function(*args, **kwargs, **_kwargs)
        
    if model_name=='base':
        pass
    elif model_name=='lab':
        pass
    elif model_name=='dense':
        pass
    elif model_name=='vit':
        pass
    elif model_name=='imseq':
        criterion = lambda *args, **kwargs: sequence_loss(*args, **kwargs, **_kwargs)
        metric_function = lambda *args, **kwargs: sequence_f1(*args, **kwargs, **_kwargs)
    
    return criterion, metric_function

def get_model(config, model_name=None, **kwargs):
    if model_name=='base':
        return CNN2RNN(max_len=config.MAX_LEN, embedding_dim=config.EMBEDDING_DIM, \
                        num_features=config.NUM_FEATURES, class_n=config.CLASS_N, \
                        rate=config.DROPOUT_RATE)
    elif model_name=='lab':
        return LAB_model(config)
    elif model_name=='dense':
        return DenseNet(config)
    elif model_name=='vit':
        return MyViT(config)
    elif model_name=='imseq':
        return ImToSeqTransformer(config)

def get_scheduler(optimizer, sch_name='none', lr=0):
    if sch_name == 'none':
        return None
    elif sch_name == 'reduce':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5,
                                                           patience=3, mode='min')
    elif sch_name == 'cosine':
        return CosineAnnealingWarmUpRestarts(optimizer, T_0=25, T_mult=1,
                                             eta_max=lr, T_up=5, gamma=0.8)

def scheduler_step(scheduler, sch_name='none', epoch=None, value=None):
    if sch_name == 'none':
        pass
    elif sch_name == 'reduce':
        scheduler.step(value)
    elif sch_name == 'cosine':
        scheduler.step(epoch)

def save_epoch(config, epoch, model, optimizer, scheduler, hist):
    torch.save(model, f'{config.SAVE_PATH}/model_{epoch+1}.pt')
    torch.save(hist, f'{config.SAVE_PATH}/train_history.pt')
    if scheduler:
        torch.save(scheduler.state_dict(), f'{config.SAVE_PATH}/scheduler_states.pt')
        torch.save(optimizer.state_dict(),
                    f'{config.SAVE_PATH}/optimizer_states.pt')

import unicodedata

def fill_str_with_space(input_s="", max_size=40, fill_char="*", align='right'):
    """
    - ????????? ??? ????????? 2????????? ????????????, ????????? 1????????? ?????????. 
    - ?????? ??????(max_size)??? 40??????, input_s??? ?????? ????????? ????????? ????????? 
    ?????? ????????? fill_char??? ?????????.
    """
    l = 0 
    for c in input_s:
        if unicodedata.east_asian_width(c) in ['F', 'W']:
            l+=2
        else: 
            l+=1
    if align=='right':
        return input_s+fill_char*(max_size-l)
    elif align=='left':
        return fill_char*(max_size-l)+input_s
    elif align=='center':
        return fill_char*(max_size//2)+input_s+fill_char*(max_size//2)
    

def visualize_score(num_class, accuracy, total, msg:str):
    print(f'###########################################################################################################################################')
    print(f'########################################################   {msg:^20s}   ###################################################################')
    print(f'#-----------------------------------------------------------------------------------------------------------------------------------------#')
    for i in range(num_class):
        _start = '#' if i%3==0 else ''
        _end = '||' if i%3!=2 else '#\n'
        print(f"{_start} {fill_str_with_space(LABEL_KOR[i],max_size=30,fill_char=' ',align='right')} : {accuracy[i]:0.2f}  {total[i]:5.0f} ", end=_end)
    print()
    print(f'############################################################################################################################################')      
    
def main(args):
    config = OmegaConf.load(f'config/config.yaml')
    mconfig = OmegaConf.load(f'config/{args.model_name}_config.yaml')
    
    TRAIN = mconfig.TRAIN
    DATA  = config.DATA
    
    IMAGE_PATH = DATA.DATA_ROOT + '/' + DATA.IMAGE_PATH
    TRAIN_PATH = DATA.DATA_ROOT + '/' + DATA.TRAIN_PATH
    VALID_PATH = DATA.DATA_ROOT + '/' + DATA.VALID_PATH
    
    ##########################      DataLoader ??????     #########################
    print('Data Loading...')
    
    preprocessor = get_preprocessor(TRAIN, args.model_name)
        
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
        train_dataloader = DataLoader(train_dataset, batch_size=TRAIN.BATCH_SIZE,
                                      num_workers=config.TRAIN.NUM_WORKER, shuffle=True)
    else:
        with open(TRAIN_PATH, 'r') as f:
            train_dataset = CustomDataset(f.read().split('\n'), pre=preprocessor)
        with open(VALID_PATH, 'r') as f:
            val_dataset = CustomDataset(f.read().split('\n'), pre=preprocessor)
            
        train_dataloader = DataLoader(train_dataset, batch_size=TRAIN.BATCH_SIZE, 
                                      num_workers=config.TRAIN.NUM_WORKER, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=TRAIN.BATCH_SIZE, 
                                    num_workers=config.TRAIN.NUM_WORKER, shuffle=False)
    ##################            Define metrics          #######################
    
    criterion, metric_function = get_metrics(args.model_name, smoothing=args.smoothing, gamma=args.gamma)
    
    ################  Model / Optimizer / Scheduler ??????  ################
    print('Model Loading...')
    
    if args.from_epoch:
        assert os.path.exists(TRAIN.SAVE_PATH + '/' + f'model_{args.from_epoch}.pt'), f'Model is not Exists: {TRAIN.SAVE_PATH}/model_{args.from_epoch}.pt'
        model = torch.load(TRAIN.SAVE_PATH + '/' + f'model_{args.from_epoch}.pt')
    else:
        # Argument??? ?????? model ??????
        model = get_model(TRAIN, model_name=args.model_name)

    print('MODEL PARAMETER # :', sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN.LEARNING_RATE)
    if args.scheduler == 'cosine':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-8) # lr should be very small value
        
    scheduler = get_scheduler(optimizer, args.scheduler, lr=TRAIN.LEARNING_RATE)
    
    if args.from_epoch and os.path.exists(f'{TRAIN.SAVE_PATH}/optimizer_states.pt'):
        optimizer.load_state_dict(torch.load(f'{TRAIN.SAVE_PATH}/optimizer_states.pt'))
        scheduler.load_state_dict(torch.load(f'{TRAIN.SAVE_PATH}/scheduler_states.pt'))
    
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
        correct_per_class = np.zeros(TRAIN.CLASS_N)
        total_per_class = np.zeros(TRAIN.CLASS_N)
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_acc, batch_correct, batch_total = train_step(model, criterion, optimizer, batch_item, 
                                                                           preprocess=preprocessor, metric_function=metric_function, num_class=len(correct_per_class))
            total_loss += batch_loss
            total_acc += batch_acc
            
            correct_per_class += batch_correct
            total_per_class += batch_total
                
            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Loss': '{:06f}'.format(batch_loss.item()),
                'Mean Loss' : '{:06f}'.format(total_loss/(batch+1)),
                'Mean F-1' : '{:06f}'.format(total_acc/(batch+1)),
                'Learning_rate' : '{}'.format(optimizer.param_groups[0]['lr'])
            })
        train_batch = batch
        
        acc_per_class = correct_per_class / total_per_class
        
        if args.for_submission:
            total_val_loss = torch.tensor([0])
            total_val_acc = 0
            val_batch = 0
        else:
            tqdm_dataset = tqdm(enumerate(val_dataloader))
            val_correct_per_class = np.zeros(TRAIN.CLASS_N)
            val_total_per_class = np.zeros(TRAIN.CLASS_N)
            for batch, batch_item in tqdm_dataset:
                batch_loss, batch_acc, batch_correct, batch_total = valid_step(model, criterion, optimizer, batch_item, 
                                                                               preprocess=preprocessor, metric_function=metric_function, num_class=len(correct_per_class))
                total_val_loss += batch_loss
                total_val_acc += batch_acc
                
                val_correct_per_class += batch_correct
                val_total_per_class += batch_total
                
                tqdm_dataset.set_postfix({
                    'Epoch': epoch + 1,
                    'Val Loss': '{:06f}'.format(batch_loss.item()),
                    'Mean Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                    'Mean Val F-1' : '{:06f}'.format(total_val_acc/(batch+1))
                })
            
            val_acc_per_class = val_correct_per_class / val_total_per_class
                
            val_batch = batch
        
        scheduler_step(scheduler, args.scheduler, 
                       value = total_val_loss/(val_batch+1), # total_val_acc/(val_batch+1),
                       epoch = epoch)
        
        convergence = scheduler.get_lr()[0]==1e-8
        
        if len(loss_plot)==epoch:
            loss_plot.append(total_loss.item()/(train_batch+1))
            metric_plot.append(total_acc/(train_batch+1))
            val_loss_plot.append(total_val_loss.item()/(val_batch+1))
            val_metric_plot.append(total_val_acc/(val_batch+1))
        else:
            loss_plot[epoch]=total_loss.item()/(train_batch+1)
            metric_plot[epoch]=total_acc/(train_batch+1)
            val_loss_plot[epoch]=total_val_loss.item()/(val_batch+1)
            val_metric_plot[epoch]=total_val_acc/(val_batch+1)
                
        # check/make directory for model file
        dp = []
        for directory in TRAIN.SAVE_PATH.split('/'):
            if directory:
                new_dir = '/'.join([*dp, directory])
                if not os.path.exists(new_dir):
                    os.mkdir(new_dir)
            dp.append(directory)
        
        
        # if total_val_acc/(val_batch+1) == max(val_metric_plot):
        #     torch.save(model, f'{TRAIN.SAVE_PATH}/model_best_f1.pt')
        if total_val_loss/(val_batch+1) == min(val_loss_plot):
            torch.save(model, f'{TRAIN.SAVE_PATH}/model_best.pt')
            
            # Visualize Result
            if total_val_acc/(val_batch+1) > 0.8:
                visualize_score(TRAIN.CLASS_N, acc_per_class, total_per_class, 'TRAIN ACCURACY SCORE')
                if val_batch:
                    visualize_score(TRAIN.CLASS_N, val_acc_per_class, val_total_per_class, 'VALID ACCURACY SCORE')
        
        if ((epoch+1) % config.TRAIN.SAVE_PERIOD == 0) or (epoch+1 == TRAIN.EPOCHS) or convergence:
            save_epoch(TRAIN, epoch, model, optimizer, scheduler, 
                       {'train_loss':loss_plot, 'val_loss':val_loss_plot,
                        'train_f1':metric_plot, 'val_f1':val_metric_plot}
                       )
            
if __name__=='__main__':
    
    # Load Config
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--from_epoch',
                        type=int, default=0)
    
    parser.add_argument('-m', '--model_name',
                        type=str, default='base')
    parser.add_argument('-sch', '--scheduler',
                        type=str, default='none')
    
    parser.add_argument('-smooth','--smoothing',
                        type=float, default=0)
    parser.add_argument('-y','--gamma',
                        type=float, default=0)
    parser.add_argument('-s','--for_submission', action='store_true')
    
    parser.add_argument('-ip','--model_path', default='none')
    
    args = parser.parse_args()
    
    main(args)