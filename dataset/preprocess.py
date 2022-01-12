from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2, pickle
import torch

def refine_csv(df):
    
    for feature in df.columns:
        feat = df[feature]
        df[feature] = feat._where(~feat.isin([0,'-']),feat[~feat.isin([0,'-'])].astype(float).mean())
        
    return df.astype(float)

class Base_Processer():
    
    def __init__(self, config):
        self.config = config
        
        crop = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}
        disease = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}
        risk = {'1':'초기','2':'중기','3':'말기'}

        self.label_description = {}
        for key, value in disease.items():
            self.label_description[f'{key}_00_0'] = f'{crop[key]}_정상'
            for disease_code in value:
                for risk_code in risk:
                    label = f'{key}_{disease_code}_{risk_code}'
                    self.label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'

        self.csv_feature_dict=None    
        self.label_dict = {key:idx for idx, key in enumerate(self.label_description)}
        
        
    def label_encoder(self, label):
        return torch.tensor(self.label_dict[label], dtype=torch.long)
    
    @property
    def label_decoder(self):
        return {val:key for key, val in self.label_dict.items()}
        
    def init_csv(self):
        config = self.config

        # 분석에 사용할 feature 선택
        csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고', 
                        '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']

        image_folder = f'{config.DATA.DATA_ROOT}/{config.DATA.IMAGE_PATH}'
        csv_files = sorted(glob(image_folder + '/*/*.csv'))

        temp_csv = pd.read_csv(csv_files[0])[csv_features]
        max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

        # feature 별 최대값, 최솟값 계산
        for csv in tqdm(csv_files[1:]):
            temp_csv = pd.read_csv(csv)[csv_features]
            
            temp_csv = refine_csv(temp_csv)
            
            temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
            max_arr = np.max([max_arr,temp_max], axis=0)
            min_arr = np.min([min_arr,temp_min], axis=0)

        # feature 별 최대값, 최솟값 dictionary 생성
        self.csv_feature_dict = {csv_features[i]:[min_arr[i], max_arr[i]] for i in range(len(csv_features))}
        with open(f'{config.DATA.DATA_ROOT}/base_pre_dict.pkl', 'wb') as f:
            pickle.dump({'csv_feature_dict':self.csv_feature_dict, 
                        'label_dict':self.label_dict}, f)

    def img_preprocessing(self, img):
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)/255
        img = np.transpose(img, (2,0,1))
        return img

    def json_preprocessing(self, json_dic):
        annotations = []
        
        for annot in json_dic['annotations']['part']:
            #(X,Y,W,H)
            annotations.append(f"{annot['x']} {annot['y']} {annot['w']} {annot['h']}\n")
            
        return annotations
        
    def csv_preprocessing(self, df):
        config=self.config
        df = df.copy()
        df = df[self.csv_feature_dict.keys()]
        df = refine_csv(df)
                
        # MinMax scaling
        for col in self.csv_feature_dict.keys():
            df[col] = df[col] - self.csv_feature_dict[col][0]
            df[col] = df[col] / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])
        # zero padding
        pad = np.zeros((config.TRAIN.MAX_LEN, len(df.columns)))
        length = min(config.TRAIN.MAX_LEN, len(df))
        pad[-length:] = df.to_numpy()[-length:]
        # transpose to sequential data
        csv_feature = pad.T
        
        return csv_feature
    
    def load_dictionary(self, path):
        with open(path, 'rb') as f:
            dict = pickle.load(f)
        self.csv_feature_dict = dict['csv_feature_dict']
        self.label_dict    = dict['label_dict']
        
class JK_Processer():
    
    def __init__(self, config):
        self.config = config
        self.label_description = {
            'crop': {'1':'딸기', '2':'토마토', '3':'파프리카',
                     '4':'오이', '5':'고추', '6':'시설포도'},
            'disease': {'00':'정상', 
                        'a1':'딸기잿빛곰팡이병', 'a2':'딸기흰가루병', 'a3':'오이노균병', 'a4':'오이흰가루병', 
                        'a5':'토마토흰가루병', 'a6':'토마토잿빛곰팡이병', 'a7':'고추탄저병', 'a8':'고추흰가루병', 
                        'a9':'파프리카흰가루병', 'a10':'파프리카잘록병', 'a11':'시설포도탄저병', 'a12':'시설포도노균병',
                        'b1':'냉해피해', 'b2':'열과', 'b3':'칼슘결핍', 'b4':'일소피해', 'b5':'축과병', 
                        'b6':'다량원소결핍 (N)', 'b7':'다량원소결핍 (P)', 'b8':'다량원소결핍 (K)'},
                        # 'c1':'딸기잿빛곰팡이병반응', 'c2':'딸기흰가루병반응', 'c3':'오이노균병반응', 'c4':'오이흰가루병반응', 
                        # 'c5':'토마토흰가루병반응', 'c6':'토마토잿빛곰팡이병반응', 'c7':'고추탄저병반응', 'c8':'고추흰가루병반응', 
                        # 'c9':'파프리카흰가루병반응', 'c10':'파프리카잘록병반응', 'c11':'시설포도탄저병반응', 'c12':'시설포도노균병반응',
            'risk': {'0':'정상', '1':'초기', '2':'중기', '3':'말기'},
            }

        self.csv_feature_dict=None    
        self.crop_encoder = {key:idx for idx, key in enumerate(self.label_description['crop'])}
        self.disease_encoder = {key:idx for idx, key in enumerate(self.label_description['disease'])}
        self.risk_encoder = {key:idx for idx, key in enumerate(self.label_description['risk'])}
        
    def label_encoder(self, label):
        crop, disease, risk = label.split('_')
        return (torch.tensor(self.crop_encoder[crop], dtype=torch.long),
                torch.tensor(self.disease_encoder[disease], dtype=torch.long),
                torch.tensor(self.risk_encoder[risk], dtype=torch.long),)
        
    def init_csv(self):
        config = self.config

        # 분석에 사용할 feature 선택
        csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고', 
                        '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']

        image_folder = f'{config.DATA.DATA_ROOT}/{config.DATA.IMAGE_PATH}'
        csv_files = sorted(glob(image_folder + '/*/*.csv'))

        temp_csv = pd.read_csv(csv_files[0])[csv_features]
        max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

        # feature 별 최대값, 최솟값 계산
        for csv in tqdm(csv_files[1:]):
            temp_csv = pd.read_csv(csv)[csv_features]
            temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
            max_arr = np.max([max_arr,temp_max], axis=0)
            min_arr = np.min([min_arr,temp_min], axis=0)

        # feature 별 최대값, 최솟값 dictionary 생성
        self.csv_feature_dict = {csv_features[i]:[min_arr[i], max_arr[i]] for i in range(len(csv_features))}
        with open(f'{config.DATA.DATA_ROOT}/drj_pre_dict.pkl', 'wb') as f:
            pickle.dump({'csv_feature_dict':self.csv_feature_dict, 
                        'crop_encoder':self.crop_encoder,
                        'disease_encoder':self.disease_encoder, 
                        'risk_encoder':self.risk_encoder}, f)

    def img_preprocessing(self, img):
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)/255
        img = np.transpose(img, (2,0,1))
        return img

    def json_preprocessing(self, json_dic):
        annotations = []
        
        for annot in json_dic['annotations']['part']:
            #(X,Y,W,H)
            annotations.append(f"{annot['x']} {annot['y']} {annot['w']} {annot['h']}")
            
        return annotations
        
    def csv_preprocessing(self, df):
        config=self.config
        df = df.copy()
                
        # MinMax scaling
        for col in self.csv_feature_dict.keys():
            df[col] = df[col] - self.csv_feature_dict[col][0]
            df[col] = df[col] / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])
        
        # transpose to sequential data
        csv_feature = df[self.csv_feature_dict.keys()].to_numpy()[-config.TRAIN.MAX_LEN:].T
        
        return csv_feature
    
    def load_dictionary(self, path):
        with open(path, 'rb') as f:
            dict = pickle.load(f)
        self.csv_feature_dict = dict['csv_feature_dict']
        self.crop_encoder     = dict['crop_encoder']
        self.disease_encoder  = dict['disease_encoder']
        self.risk_encoder     = dict['risk_encoder']
        
class Dense_Processer():
    
    def __init__(self, config):
        self.config = config
        
        crop = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}
        disease = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}
        risk = {'1':'초기','2':'중기','3':'말기'}

        self.label_description = {}
        for key, value in disease.items():
            self.label_description[f'{key}_00_0'] = f'{crop[key]}_정상'
            for disease_code in value:
                for risk_code in risk:
                    label = f'{key}_{disease_code}_{risk_code}'
                    self.label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'

        self.csv_feature_dict=None    
        self.label_dict = {key:idx for idx, key in enumerate(self.label_description)}
        
        
    def label_encoder(self, label):
        return torch.tensor(self.label_dict[label], dtype=torch.long)
    
    @property
    def label_decoder(self):
        return {val:key for key, val in self.label_dict.items()}
        
    def init_csv(self):
        config = self.config

        # 분석에 사용할 feature 선택
        csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고', 
                        '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']

        image_folder = f'{config.DATA.DATA_ROOT}/{config.DATA.IMAGE_PATH}'
        csv_files = sorted(glob(image_folder + '/*/*.csv'))

        temp_csv = pd.read_csv(csv_files[0])[csv_features]
        max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

        # feature 별 최대값, 최솟값 계산
        for csv in tqdm(csv_files[1:]):
            temp_csv = pd.read_csv(csv)[csv_features]
            
            temp_csv = refine_csv(temp_csv)
            
            temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
            max_arr = np.max([max_arr,temp_max], axis=0)
            min_arr = np.min([min_arr,temp_min], axis=0)

        # feature 별 최대값, 최솟값 dictionary 생성
        self.csv_feature_dict = {csv_features[i]:[min_arr[i], max_arr[i]] for i in range(len(csv_features))}
        with open(f'{config.DATA.DATA_ROOT}/dense_pre_dict.pkl', 'wb') as f:
            pickle.dump({'csv_feature_dict':self.csv_feature_dict, 
                        'label_dict':self.label_dict}, f)

    def img_preprocessing(self, img):
        img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)/255
        img = np.transpose(img, (2,0,1))
        return img

    def json_preprocessing(self, json_dic):
        annotations = []
        
        for annot in json_dic['annotations']['part']:
            #(X,Y,W,H)
            annotations.append(f"{annot['x']} {annot['y']} {annot['w']} {annot['h']}\n")
            
        return annotations
        
    def csv_preprocessing(self, df):
        config=self.config
        df = df.copy()
        df = df[self.csv_feature_dict.keys()]
        df = refine_csv(df)
                
        # MinMax scaling
        for col in self.csv_feature_dict.keys():
            df[col] = df[col] - self.csv_feature_dict[col][0]
            df[col] = df[col] / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])
        # zero padding
        pad = np.zeros((config.TRAIN.MAX_LEN, len(df.columns)))
        length = min(config.TRAIN.MAX_LEN, len(df))
        pad[-length:] = df.to_numpy()[-length:]
        # transpose to sequential data
        csv_feature = pad.T
        
        return csv_feature
    
    def load_dictionary(self, path):
        with open(path, 'rb') as f:
            dict = pickle.load(f)
        self.csv_feature_dict = dict['csv_feature_dict']
        self.label_dict    = dict['label_dict']