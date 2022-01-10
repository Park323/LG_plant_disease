from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2, pickle

class Base_Processer():
    
    def __init__(self, config):
        self.config = config
        self.label_description = {
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
            
        self.csv_feature_dict = None
        self.label_encoder = {key:idx for idx, key in enumerate(self.label_description)}
        self.label_decoder = {val:key for key, val in self.label_encoder.items()}
    
    def img_preprocessing(self, img):
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)/255
        img = np.transpose(img, (2,0,1))
        return img

    def json_preprocessing(self, json_dic):
        return json_dic
        
    def csv_preprocessing(self, df):
        df = df.copy()
        config = self.config
        
        if self.csv_feature_dict:
            pass
        else:
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
            
            with open(f'{config.DATA.DATA_ROOT}/prepro_dict.pkl', 'wb') as f:
                pickle.dump({'csv_feature_dict':self.csv_feature_dict, 
                            'label_encoder':self.label_encoder,
                            'label_decoder':self.label_decoder}, f)
                
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
        self.label_decoder    = dict['label_decoder']
        self.label_encoder    = dict['label_encoder']