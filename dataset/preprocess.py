from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2, pickle, json
import torch
from torchvision.transforms import transforms

from dataset.csv_process import reduce_number

# area_dict = {'1':'열매','2':'꽃','3':'잎','4':'가지','5':'줄기','6':'뿌리','7':'해충'}
# grow_dict = {'11':'유묘기', '12':'생장기', '13':'착화/과실기', 
#              '21':'발아기', '22':'개화기', '23':'신초생장기',
#              '24':'과실성숙기', '25':'수확기', '26':'휴면기'}
area_dict = {'1':'열매','2':'꽃','3':'잎','4':'가지','5':'줄기'}
grow_dict = {'11':'유묘기', '12':'생장기', '13':'착화/과실기', '24':'과실성숙기'}
crop_dict = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}
# disease_dict = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},               # 1,2,13,18,19,20
#                 '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},# 5,6,14,15,18,19,20
#                 '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},             # 9,10,15,18,19,20
#                 '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},                     # 3,4,13,18,19,20
#                 '5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},                     # 7,8,15,18,19,20
#                 '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}                                                                     # 11,12,16,17
disease_dict = {'1':{},
                '2':{'a5':['2']}, #'토마토흰가루병'                                                                                     # 1
                '3':{'a9':['1','2','3'], #'파프리카흰가루병'
                     'b3':['1'], #'칼슘결핍',
                     'b6':['1'], #다량원소결핍 (N)',
                     'b7':['1'], #'다량원소결핍 (P)',
                     'b8':['1']}, #다량원소결핍 (K)'},# 3,6,9,10,11
                '4':{},
                '5':{'a7':['2'], #'고추탄저병',
                     'b6':['1'], #'다량원소결핍 (N)',
                     'b7':['1'], #'다량원소결핍 (P)',
                     'b8':['1']}, #'다량원소결핍 (K)'},                     # 2,9,10,11
                '6':{'a11':['1','2'], #'시설포도탄저병',
                     'a12':['1','2'], #'시설포도노균병',
                     'b4':['1','3'], #'일소피해',
                     'b5':['1']}} #'축과병'}}                                   # 4,5,7,8
disease_name = {'1':{},
                '2':{'a5':'흰가루병'},
                '3':{'a9':'흰가루병',
                     'b3':'칼슘결핍',
                     'b6':'질소결핍',
                     'b7':'인결핍',
                     'b8':'칼륨결핍'},
                '4':{},
                '5':{'a7':'탄저병',
                     'b6':'질소결핍',
                     'b7':'인결핍',
                     'b8':'칼륨결핍'},
                '6':{'a11':'탄저병',
                     'a12':'노균병',
                     'b4':'일소피해',
                     'b5':'축과병'}}
disease_dict2= {'00':'정상', 
                'a5':'토마토흰가루병', 'a7':'고추탄저병',
                'a9':'파프리카흰가루병', 'a11':'시설포도탄저병', 'a12':'시설포도노균병',
                'b3':'칼슘결핍', 'b4':'일소피해', 'b5':'축과병', 
                'b6':'다량원소결핍 (N)', 'b7':'다량원소결핍 (P)', 'b8':'다량원소결핍 (K)'}
risk_dict = {'1':'초기','2':'중기','3':'말기'}
risk_dict2 = {'0':'정상','1':'초기','2':'중기','3':'말기'}

partial_dict = {key:value for key, value in [('<S>','StartToken'),
                                             ('<E>','EndToken'),
                                             *crop_dict.items(), 
                                             *[('#'+idx, disease) for idx, disease in disease_dict2.items()], 
                                             *[('##'+idx, risk) for idx, risk in risk_dict2.items()]]}

label_description = {}
for key, value in disease_dict.items():
    label_description[f'{key}_00_0'] = f'{crop_dict[key]}_정상'
    for disease_code, risk_list in value.items():
        for risk_code in risk_list:
            label = f'{key}_{disease_code}_{risk_code}'
            label_description[label] = f'{crop_dict[key]}_{disease_name[key][disease_code]}_{risk_dict[risk_code]}'

def refine_csv(df):
    
    for feature in df.columns:
        feat = df[feature]
        df[feature] = feat._where(~feat.isin([0,'-']),feat[~feat.isin([0,'-'])].astype(float).mean())
        
    return df.astype(float)

class Processor():
    def __init__(self, config):
        self.config = config
        
        self.dict_name = 'none'
        self.feature_dict = {}
        
        self.area_dict = {key:idx for idx, key in enumerate(area_dict)}
        self.grow_dict = {key:idx for idx, key in enumerate(grow_dict)}
        self.crop_dict = {key:idx for idx, key in enumerate(crop_dict)}
        self.disease_dict = {key:idx for idx, key in enumerate(disease_dict2)}
        self.risk_dict = {key:idx for idx, key in enumerate(risk_dict)}
        self.label_dict = {key:idx for idx, key in enumerate(label_description)}
        self.partial_dict = {key:idx for idx, key in enumerate(partial_dict)}
        
        self.img_transforms = transforms.Compose([
        ])
        
    def label_encoder(self, label, *args, **kwargs):
        pass
    
    def label_decoder(self, label, *args, **kwargs):
        pass
    
    def label_to_index(self, label, *args, **kwargs):
        pass
    
    def initialize(self):
        pass

    def img_processing(self, img, Train=True, **kwargs):
        img = transforms.ToTensor()(img)
        if Train:
            img = self.img_transforms(img)
        img = transforms.Resize((224,224))(img)
        img = img/255
        img = transforms.Normalize(img.mean(), img.std())(img)
        return img

    def json_processing(self, json_file, **kwargs):
        return np.array([[0]])
        
    def csv_processing(self, df, **kwargs):
        return np.array([[0]])
    
    def save_dictionary(self):
        with open(f'{self.config.DATA.DATA_ROOT}/{self.dict_name}', 'wb') as f:
            pickle.dump(self.feature_dict, f)
            
    def load_dictionary(self, path):
        with open(path, 'rb') as f:
            dic = pickle.load(f)
        self.feature_dict = dic

class Base_Processor(Processor):
    '''
    This Processor Generates Labels with string with proper pair (CLASS_N=111)
    '''
    def __init__(self, config):
        super(Base_Processor, self).__init__(config)
        
    def label_encoder(self, label, *args, **kwargs):
        return torch.tensor(self.label_dict[label], dtype=torch.long)
    
    def label_decoder(self, label, *args, **kwargs):
        return {val:key for key, val in self.label_dict.items()}[label]
    
    def partial_encoder(self, label, *args, **kwargs):
        return torch.tensor(self.partial_dict[label], dtype=torch.long)
    
    def partial_decoder(self, label, *args, **kwargs):
        return {val:key for key, val in self.partial_dict.items()}[label]
    
class TrimmedImage_Processor(Base_Processor):
    def img_processing(self, img):
        
        return super().img_processing(img)

class Basic_CSV_Processor(Base_Processor):
    '''
    This Processor returns csv through min-max scaling
    '''
    def __init__(self, config):
        super(Basic_CSV_Processor, self).__init__(config)
        self.dict_name = 'csv_feature_dict.pkl'
      
    def initialize(self):  
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
        self.feature_dict['csv_feature_dict'] = {csv_features[i]:[min_arr[i], max_arr[i]] for i in range(len(csv_features))}
        
        self.save_dictionary()
        
    def csv_processing(self, df):
        config=self.config
        csv_feature_dict = self.feature_dict['csv_feature_dict']
        df = df.copy()
        df = df[csv_feature_dict.keys()]
        df = refine_csv(df)
                
        # MinMax scaling
        for col in csv_feature_dict.keys():
            df[col] = df[col] - csv_feature_dict[col][0]
            df[col] = df[col] / (csv_feature_dict[col][1]-csv_feature_dict[col][0])
        # zero padding
        pad = np.zeros((config.MAX_LEN, len(df.columns)))
        length = min(config.MAX_LEN, len(df))
        pad[-length:] = df.to_numpy()[-length:]
        # transpose to sequential data
        csv_feature = pad.T
        
        return csv_feature

class Concat_Processor(Processor):
    '''
    This Processor Generates Labels with concatenated one_hot_label
    ex) return tensor [...1...|...1...|...1...|...1...] with size(B, 38)
        crop:6, disease:21, risk:4, area:7
    '''
    
    def __init__(self, config):
        super(Concat_Processor, self).__init__(config)
        
    def label_encoder(self, label, *args, **kwargs):
        area = str(kwargs['dic']['annotations']['area'])
        # grow = str(kwargs['dic']['annotations']['grow'])
        
        crop, disease, risk = label.split('_')
        
        # one_hot_label = torch.zeros(47, dtype=torch.float32)
        one_hot_label = torch.zeros(38, dtype=torch.float32)
        one_hot_label[self.crop_dict[crop]] = 1.
        one_hot_label[6+self.disease_dict[disease]] = 1.
        one_hot_label[6+21+self.risk_dict[risk]] = 1.
        one_hot_label[6+21+4+self.area_dict[area]] = 1.
        # one_hot_label[6+21+4+7+self.grow_dict[grow]] = 1.
            
        # epsilon = 0.1
        # smoothing = lambda x: (1-epsilon)*x + epsilon/len(x)
        
        # one_hot_label = smoothing(one_hot_label)
        
        return one_hot_label
    
    def label_decoder(self, label, *args, **kwargs):
        crop = {val:key for key, val in self.crop_dict.items()}[label[0]]
        disease = {val:key for key, val in self.disease_dict.items()}[label[1]]
        risk = {val:key for key, val in self.risk_dict.items()}[max(0,min(3,round(label[2])))]
        
        if disease not in disease_dict[crop].keys() or ((disease=='00') != (risk=='0')):
            return f'{crop}_00_0'
        return f'{crop}_{disease}_{risk}'
        
    def initialize(self):
        self.save_dictionary()

class CropOnly_Processor(Processor):
    '''
    This Processor Generates Labels with one_hot_label of crop
    ex) return tensor [...1...|...1...|...1...|...1...] with size(B, 38)
        crop:6, disease:21, risk:4, area:7
    '''
    
    def __init__(self, config):
        super(CropOnly_Processor, self).__init__(config)
        
    def label_encoder(self, label, *args, **kwargs):
        area = str(kwargs['dic']['annotations']['area'])
        
        crop, disease, risk = label.split('_')
        
        one_hot_label = torch.zeros(6, dtype=torch.float32)
        one_hot_label[self.crop_dict[crop]] = 1.
        # one_hot_label[6+self.area_dict[area]] = 1.
        
        return one_hot_label
    
    def label_decoder(self, label, *args, **kwargs):
        crop = {val:key for key, val in self.crop_dict.items()}[label[0]]
        return f'{crop}'
        
    def initialize(self):
        self.save_dictionary()

class Dense_Processor(Base_Processor):
    def __init__(self, config):
        super().__init__(config)
        
class ViT_Processor(Base_Processor):
    def __init__(self, config):
        super().__init__(config)
        self.USE_SPOT = config.USE_SPOT
        self.img_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3,
                                   contrast=0.3,
                                   saturation=0.3),
            transforms.RandomAffine(180, shear=10) 
                                    # interpolation=transforms.InterpolationMode.BILINEAR)
        ]) if config.USE_AUG else transforms.Compose([])
        self.color_scale = lambda x: x
        self.csv_transforms = lambda x : self.temporal_drop(x, drop_rate=config.CSV_DROP_RATE)
        
    def replace_by_mean(self, arr, replace_rate=0.3):
        cols = [
            '내부 온도 평균', '내부 온도 최고', '내부 온도 최저',
            '내부 습도 평균', '내부 습도 최고', '내부 습도 최저',
            '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저', 
            '내부 CO2 평균', '내부 CO2 최고',
            '내부 CO2 최저', '외부 누적일사 평균',
            ]
        col_dic = {idx:key for idx, key in enumerate(cols)}
        
        stats = pd.read_csv('dataset/csv_statistics.csv', index_col=0)
        _stats = reduce_number(stats['mean'])
        _stats.name = 'mean'
        stats = pd.concat([stats['mean'], _stats]).to_frame()
        
        mask = np.where(np.random.rand(arr.shape[0]) < replace_rate)
        col_mask = [col_dic[idx] for idx in mask[0]]
        arr[mask] = stats.loc[col_mask, 'mean'].to_numpy().reshape(-1,1)
        
        return arr
        
    def temporal_drop(self, arr, drop_rate=0.3):
        if drop_rate:
            if np.random.rand() < drop_rate:    
                arr[:,np.random.randint(1,self.config.MAX_LEN):] = 0
        return arr
        
    def csv_processing(self, df, **kwargs):
        feat = df.iloc[:,1:].to_numpy()
        feat = feat.T
        # Pad
        outputs = np.zeros((feat.shape[0],self.config.MAX_LEN))
        outputs[:,:feat.shape[1]] = feat[:,:self.config.MAX_LEN]
        # Augment
        outputs = self.csv_transforms(outputs)
        return outputs
    
    def img_processing(self, img, Json=-1, Train=True, **kwargs):
        if self.USE_SPOT:
            img = np.concatenate([img, Json], axis=-1)
        img = transforms.ToTensor()(img)
        img = transforms.CenterCrop((512,736))(img)
        img = transforms.Resize((256,368))(img)
        img = self.color_scale(img)
        if Train:
            img[:3] = self.img_transforms(img[:3]) # Only for color image (not alpha channel)
        img = transforms.Normalize(img.mean(), img.std())(img)
        return img
    
    def json_processing(self, json_file, image_size=None, **kwargs):
        spot_channel = np.zeros((*image_size[:2],1))
        spot = json_file['annotations']['bbox'][0]
        x, y, w, h = list(map(int, [spot['x'], spot['y'], spot['w'], spot['h']]))
        spot_channel[y:y+h, x:x+w] = 255.
        # for spot in json_file['annotations']['part']:
        #     x, y, w, h = list(map(int, [spot['x'], spot['y'], spot['w'], spot['h']]))
        #     spot_channel[y:y+h, x:x+w] = 255.
        return spot_channel
        
class LAB_Processor(ViT_Processor):
    def __init__(self, config):
        super().__init__(config)
        self.color_scale = self.rgb2lab
        
    def rgb2lab(self, image):
        image = image.permute(1,2,0)
        image[:,:,:3] = self.xyz2lab(self.rgb2xyz(image[:,:,:3].clone()))
        return image.permute(2,0,1)
    
    def rgb2xyz(self, img):
        # https://www.easyrgb.com/en/math.php
        # input is numpy (B, W, H, C)
        # sR, sG and sB (Standard RGB) input range = 0 ÷ 255
        # X, Y and Z output refer to a D65/2° standard illuminant.
        
        mask = img > 0.04045
        img[mask] = np.power((img[mask] + 0.055) / 1.055, 2.4)
        img[~mask] /= 12.92
        
        img *= 100
        
        xyz_conv = np.array([[0.4124, 0.3576, 0.1805],
                            [0.2126, 0.7152, 0.0722],
                            [0.0193, 0.1192, 0.9505]])
        
        return img @ xyz_conv.T

    def xyz2lab(self, img):
        # https://www.easyrgb.com/en/math.php
        # input is tensor (B, C, W, H)
        # Reference-X, Y and Z refer to specific illuminants and observers.
        # Common reference values are available below in this same page.
        refX, refY, refZ = 0.95047, 1., 1.08883   # This was: `lab_ref_white` D65 / 2
        
        img[:,:,0] /= refX
        img[:,:,1] /= refY
        img[:,:,2] /= refZ

        mask = img > 0.008856
        
        img[mask] = np.power(img[mask], 1/3)
        img[~mask] = 7.787 * img[~mask] + 16/116

        lab_conv = np.array([[0,    116,    0],
                            [500, -500,    0],
                            [0,    200, -200]])

        img = img @ lab_conv.T + np.array([-16, 0, 0])
        
        # import matplotlib.pyplot as plt
        # plt.imshow(img[0,:,:,0], cmap='gray')
        # plt.show()
        # plt.imshow(img[0,:,:,1], cmap='gray')
        # plt.show()
        # plt.imshow(img[0,:,:,2], cmap='gray')
        # plt.show()
        
        return img
    
    
class Seq_Processor(Base_Processor):
    def __init__(self, config):
        super().__init__(config)
        
        self.img_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3,
                                   contrast=0.3,
                                   saturation=0.3),
            transforms.RandomAffine(180, shear=10) 
                                    # interpolation=transforms.InterpolationMode.BILINEAR)
        ])
    
    def super_encoder(self, label, *args, **kwargs):
        return super().label_encoder(label, *args, **kwargs)
        
    def label_encoder(self, label, *args, **kwargs):
        c, d, r = label.split('_')
        label  = torch.tensor([self.partial_encoder('<S>'), 
                               self.partial_encoder(c), self.partial_encoder('#'+d), self.partial_encoder('##'+r), 
                               self.partial_encoder('<E>')], dtype=torch.long)
        return label
    
    def img_processing(self, img, Train=True, **kwargs):
        img = transforms.ToTensor()(img)
        img = transforms.CenterCrop((512,736))(img)
        img = transforms.Resize((224,224))(img)
        if Train:
            img = self.img_transforms(img)
        img = transforms.Normalize(img.mean(), img.std())(img)
        return img