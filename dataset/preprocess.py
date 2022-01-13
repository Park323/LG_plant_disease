from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2, pickle, json
import torch

crop_dict = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}
disease_dict = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
                '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}
disease_dict2= {'00':'정상', 
                'a1':'딸기잿빛곰팡이병', 'a2':'딸기흰가루병', 'a3':'오이노균병', 'a4':'오이흰가루병', 
                'a5':'토마토흰가루병', 'a6':'토마토잿빛곰팡이병', 'a7':'고추탄저병', 'a8':'고추흰가루병', 
                'a9':'파프리카흰가루병', 'a10':'파프리카잘록병', 'a11':'시설포도탄저병', 'a12':'시설포도노균병',
                'b1':'냉해피해', 'b2':'열과', 'b3':'칼슘결핍', 'b4':'일소피해', 'b5':'축과병', 
                'b6':'다량원소결핍 (N)', 'b7':'다량원소결핍 (P)', 'b8':'다량원소결핍 (K)'}
risk_dict = {'0':'정상','1':'초기','2':'중기','3':'말기'}

label_description = {}
for key, value in disease_dict.items():
    label_description[f'{key}_00_0'] = f'{crop_dict[key]}_정상'
    for disease_code in value:
        for risk_code in risk_dict:
            label = f'{key}_{disease_code}_{risk_code}'
            label_description[label] = f'{crop_dict[key]}_{disease_dict[key][disease_code]}_{risk_dict[risk_code]}'

def refine_csv(df):
    
    for feature in df.columns:
        feat = df[feature]
        df[feature] = feat._where(~feat.isin([0,'-']),feat[~feat.isin([0,'-'])].astype(float).mean())
        
    return df.astype(float)

class Base_Processor():
    
    def __init__(self, config):
        self.config = config
        self.dict_name = 'base_pre_dict.pkl'
        
        self.label_dict = {key:idx for idx, key in enumerate(label_description)}
        
        self.csv_feature_dict=None    
        
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
        
        self.save_dictionary()
                        
    def save_dictionary(self):
        with open(f'{self.config.DATA.DATA_ROOT}/{self.dict_name}', 'wb') as f:
            pickle.dump({'csv_feature_dict':self.csv_feature_dict, 
                        'label_dict':self.label_dict}, f)

    def img_processing(self, img):
        img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)/255
        img = np.transpose(img, (2,0,1))
        return img

    def json_processing(self, json_path):
        return json_path
        
    def csv_processing(self, df):
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
        
class Base_Processor2(Base_Processor):
    
    ###### Crop/Disease/risk 분리.
    
    def __init__(self, config):
        super(Base_Processor2, self).__init__(config)
        
        self.config = config
        self.dict_name = 'base2_pre_dict.pkl'

        self.csv_feature_dict=None    
        self.crop_dict = {key:idx for idx, key in enumerate(crop_dict)}
        self.disease_dict = {key:idx for idx, key in enumerate(disease_dict2)}
        self.risk_dict = {key:idx for idx, key in enumerate(risk_dict)}
        self.label_dict = {key:idx for idx, key in enumerate(label_description)}
        
    def label_encoder(self, label):
        crop, disease, risk = label.split('_')
        return (torch.tensor(self.crop_dict[crop], dtype=torch.long),
                torch.tensor(self.disease_dict[disease], dtype=torch.long),
                torch.tensor(self.risk_dict[risk], dtype=torch.long),)
    
    def label_decoder(self, label):
        crop = {val:key for key, val in self.crop_dict.items()}[label[0]]
        disease = {val:key for key, val in self.disease_dict.items()}[label[1]]
        risk = {val:key for key, val in self.risk_dict.items()}[label[2]]
        
        if disease not in disease_dict[crop].keys() or ((disease=='00') != (risk=='0')):
            return f'{crop}_00_0'
        return f'{crop}_{disease}_{risk}'
    
    def save_dictionary(self):
        with open(f'{self.config.DATA.DATA_ROOT}/{self.dict_name}', 'wb') as f:
            pickle.dump({'csv_feature_dict':self.csv_feature_dict, 
                        'crop_dict':self.crop_dict,
                        'disease_dict':self.disease_dict, 
                        'risk_dict':self.risk_dict}, f)
    
    def load_dictionary(self, path):
        with open(path, 'rb') as f:
            dict = pickle.load(f)
        self.csv_feature_dict = dict['csv_feature_dict']
        self.crop_dict     = dict['crop_dict']
        self.disease_dict  = dict['disease_dict']
        self.risk_dict     = dict['risk_dict']
        
class Dense_Processor():
    
    def __init__(self, config):
        self.config = config
        self.dict_name = 'dense_pre_dict.pkl'
        
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

    def img_processing(self, img):
        img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)/255
        img = np.transpose(img, (2,0,1))
        return img

    def json_processing(self, json_path):
        
        with open(json_path, 'r') as f:
            json_file = json.load(f)
        
        annotations = []
        
        for annot in json_path['annotations']['part']:
            #(X,Y,W,H)
            annotations.append(f"{annot['x']} {annot['y']} {annot['w']} {annot['h']}\n")
            
        return annotations
        
    def csv_processing(self, df):
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