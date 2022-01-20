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
from model.base_model import CNN2RNN
from model.jk_model import DrJeonko
from model.lab_model import *
from model.dense_model import DenseNet
from model.vit_model import ViT_tuned, ImToSeqTransformer
from utils.scheduler import CosineAnnealingWarmUpRestarts
from utils.metric import *

import warnings
warnings.simplefilter('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_step(
    model, criterion, optimizer, 
    batch_item, training, 
    preprocess=None, metric_function=accuracy_function, **kwargs):
    
    img = batch_item['img'].to(DEVICE)
    csv_feature = batch_item['csv_feature'].to(DEVICE)
    label = batch_item['label']
    label = label.to(DEVICE)
    
    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(img, csv_feature, labels=label)
            loss = criterion(output, label, **kwargs)
        loss.backward()
        optimizer.step()
        if 'decode' in dir(model):
            model.eval()
            with torch.no_grad():
                output = model.decode(img, csv_feature)
        score = metric_function(label, output, preprocess)
        return loss, score
    else:
        model.eval()
        with torch.no_grad():
            output = model(img, csv_feature, labels=label, train=False)
            loss = criterion(output, label, **kwargs)
        if 'decode' in dir(model):
            output = model.decode(img, csv_feature)
        score = metric_function(label, output, preprocess)
        return loss, score

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

def get_preprocessor(config, model_name=None, **kwargs):
    if model_name=='base':
        return preprocess.Basic_CSV_Processor(config)
    elif model_name=='lab':
        return preprocess.Concat_processor(config)
    elif model_name=='lab_crop':
        return preprocess.Concat_processor(config)
    elif model_name=='dense':
        return preprocess.Dense_Processor(config)
    elif model_name=='drj':
        return preprocess.Base_Processor(config)
    elif model_name=='vit':
        return preprocess.ViT_Processor(config)
    elif model_name=='imseq':
        return preprocess.Seq_Processor(config)

def get_metrics(model_name):
    
    criterion = ce_loss
    metric_function = accuracy_function
    
    if model_name=='base':
        pass
    elif model_name=='lab':
        pass
    elif model_name=='lab_crop':
        pass
    elif model_name=='dense':
        pass
    elif model_name=='drj':
        pass
    elif model_name=='vit':
        pass
    elif model_name=='imseq':
        criterion = sequence_loss
        metric_function = sequence_f1
    
    return criterion, metric_function

def get_model(config, model_name=None, **kwargs):
    if model_name=='base':
        return CNN2RNN(max_len=config.MAX_LEN, embedding_dim=config.EMBEDDING_DIM, \
                        num_features=config.NUM_FEATURES, class_n=config.CLASS_N, \
                        rate=config.DROPOUT_RATE)
    elif model_name=='lab':
        return LAB_model(config)
    elif model_name=='lab_crop':
        return CropClassifier(config)
    elif model_name=='dense':
        return DenseNet(config)
    elif model_name=='drj':
        return DrJeonko(config)
    elif model_name=='vit':
        return ViT_tuned(config)
    elif model_name=='imseq':
        return ImToSeqTransformer(config)

def get_scheduler(optimizer, sch_name='none', lr=0):
    if sch_name == 'none':
        return None
    elif sch_name == 'reduce':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5,
                                                           patience=3, mode='max')
    elif sch_name == 'cosine':
        return CosineAnnealingWarmUpRestarts(optimizer, T_0=15, T_mult=2, 
                                                  eta_max=lr, T_up=3, gamma=0.5)

def scheduler_step(scheduler, sch_name='none', epoch=None, value=None):
    if sch_name == 'none':
        pass
    elif sch_name == 'reduce':
        scheduler.step(value)
    elif sch_name == 'cosine':
        scheduler.step(epoch)
    
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
    
    preprocessor = get_preprocessor(config, args.model_name)
        
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
    
    criterion, metric_function = get_metrics(args.model_name)
    
    ##################              Inference             #######################
    if args.inference:
        assert args.model_path != 'none', 'Model Path should be passed by argument.\nExample) train.py -i -ip output/sample_model.pt'
        print('Inference Start...')
        with open(TEST_PATH, 'r') as f:
            test_dataset = CustomDataset(f.read().split('\n'), pre=preprocessor, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=TEST.BATCH_SIZE, num_workers=TEST.NUM_WORKER, shuffle=False)
        
        model = torch.load(args.model_path)
        
        preds, answer = predict(TEST, model, test_dataloader)
        
        submission = pd.read_csv(f'{TEST.SAMPLE_PATH}')
        submission['label'] = metric_function(None, preds, inference=True, preprocess=preprocessor)
        submission.to_csv(f'{TRAIN.SAVE_PATH}/submission.csv', index=False)
        
        return None
    
    ################  Model / Optimizer / Scheduler 정의  ################
    print('Model Loading...')
    
    if args.from_epoch:
        assert os.path.exists(TRAIN.SAVE_PATH + '/' + f'model_{args.from_epoch}.pt'), f'Model is not Exists: {TRAIN.SAVE_PATH}/model_{args.from_epoch}.pt'
        model = torch.load(TRAIN.SAVE_PATH + '/' + f'model_{args.from_epoch}.pt')
    else:
        # Argument에 따라 model 변경
        model = get_model(TRAIN, model_name=args.model_name)

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
        train_batch = batch
        
        if args.for_submission:
            total_val_loss = 0
            total_val_acc = 0
            val_batch = 0
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
            val_batch = batch
        
        scheduler_step(scheduler, args.scheduler, 
                       value = total_val_acc/(val_batch+1),
                       epoch = epoch)
        
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
        
        
        if total_val_acc/(val_batch+1) == max(val_metric_plot):
            torch.save(model, f'{TRAIN.SAVE_PATH}/model_best_f1.pt')
        
        if total_val_loss.item()/(val_batch+1) == min(val_loss_plot):
            torch.save(model, f'{TRAIN.SAVE_PATH}/model_min_loss.pt')
        
        if ((epoch+1) % config.TRAIN.SAVE_PERIOD == 0) or (epoch+1 == TRAIN.EPOCHS):
            torch.save(model, f'{TRAIN.SAVE_PATH}/model_{epoch+1}.pt')
            torch.save({'train_loss':loss_plot, 'val_loss':val_loss_plot,
                        'train_f1':metric_plot, 'val_f1':val_metric_plot},
                        f'{TRAIN.SAVE_PATH}/train_history.pt')
            torch.save(scheduler.state_dict(),
                        f'{TRAIN.SAVE_PATH}/scheduler_states.pt')
            torch.save(optimizer.state_dict(),
                        f'{TRAIN.SAVE_PATH}/optimizer_states.pt')
            
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
    parser.add_argument('-s','--for_submission', action='store_true')
    
    parser.add_argument('-i','--inference', action='store_true')
    parser.add_argument('-ip','--model_path', default='none')
    
    args = parser.parse_args()
    
    main(args)