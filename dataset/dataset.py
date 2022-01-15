import pandas as pd
import cv2, json

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, files, labels=None, mode='train', pre=None):
        self.mode = mode
        self.files = files
        self.preprocess = pre
        self.csv_feature_check = [0]*len(self.files)
        self.csv_features = [None]*len(self.files)
        # self.max_len = -1 * 24*6
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        file = self.files[i]
        file = file.strip()
        file_name = file.split('/')[-1]
        
        json_path = f'{file}/{file_name}.json'
        image_path = f'{file}/{file_name}.jpg'
        csv_path = f'{file}/{file_name}.csv'
        
        if self.csv_feature_check[i] == 0:
            df = pd.read_csv(csv_path)
            csv_feature = self.preprocess.csv_processing(df=df)
            self.csv_features[i] = csv_feature
            self.csv_feature_check[i] = 1
        else:
            csv_feature = self.csv_features[i]
        
        if self.mode == 'train':
            with open(json_path, 'r') as f:
                json_file = json.load(f)
        
            img = cv2.imread(image_path)
            img = self.preprocess.img_processing(img=img, labels=json_file)
            
            crop = json_file['annotations']['crop']
            disease = json_file['annotations']['disease']
            risk = json_file['annotations']['risk']
            label = f'{crop}_{disease}_{risk}'
            
            return {
                'img' : torch.tensor(img, dtype=torch.float32),
                'csv_feature' : torch.tensor(csv_feature, dtype=torch.float32),
                'json_path' : json_path,
                'label' : self.preprocess.label_encoder(label, dic=json_file)
            }
        else:
            img = cv2.imread(image_path)
            img = self.preprocess.img_processing(img=img)
            
            return {
                'img' : torch.tensor(img, dtype=torch.float32),
                'csv_feature' : torch.tensor(csv_feature, dtype=torch.float32)
            }