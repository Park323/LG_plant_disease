from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2, pickle, json
import torch
from torchvision.transforms import transforms

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

    def json_processing(self, labels, **kwargs):
        return labels
        
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
        pad = np.zeros((config.TRAIN.MAX_LEN, len(df.columns)))
        length = min(config.TRAIN.MAX_LEN, len(df))
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
        
        self.img_transforms = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2,
                                   contrast=0.15,
                                   saturation=0.1),
            transforms.RandomAffine(90, shear=10, 
                                    interpolation=transforms.InterpolationMode.BILINEAR)
        ])
    
    def img_processing(self, img, Train=True, **kwargs):
        img = transforms.ToTensor()(img)
        img = transforms.CenterCrop((512,736))(img)
        img = transforms.Resize((256,368))(img)
        if Train:
            img = self.img_transforms(img)
        img = transforms.Normalize(img.mean(), img.std())(img)
        return img
        
class Seq_Processor(Base_Processor):
    def __init__(self, config):
        super().__init__(config)
        self.img_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2,
                                   contrast=0.15,
                                   saturation=0.1),
            transforms.RandomAffine(30, shear=10, 
                                    interpolation=transforms.InterpolationMode.BILINEAR)
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
        if Train:
            img = self.img_transforms(img)
        img = img/255
        img = transforms.Normalize(img.mean(), img.std())(img)
        return img