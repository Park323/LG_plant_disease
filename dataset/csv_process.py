import pandas as pd
import numpy as np
import os, shutil, argparse
from glob import glob
from tqdm import tqdm
import pickle

use_cols = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 온도 2 평균',
            '내부 온도 2 최고', '내부 온도 2 최저', '내부 온도 3 평균', '내부 온도 3 최고', '내부 온도 3 최저',
            '내부 습도 1 평균', '내부 습도 1 최고',
            '내부 습도 1 최저', '내부 습도 2 평균', '내부 습도 2 최고', '내부 습도 2 최저', '내부 습도 3 평균',
            '내부 습도 3 최고', '내부 습도 3 최저',
            '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저', '내부 CO2 평균', '내부 CO2 최고',
            '내부 CO2 최저', '외부 누적일사 평균']

temper_cols = ['내부 온도 평균', '내부 온도 최고', '내부 온도 최저', 
               '내부 습도 평균', '내부 습도 최고', '내부 습도 최저']

reduced_cols = ['내부 온도 평균', '내부 온도 최고', '내부 온도 최저',
                '내부 습도 평균', '내부 습도 최고', '내부 습도 최저', 
                '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저', '내부 CO2 평균', '내부 CO2 최고',
                '내부 CO2 최저', '외부 누적일사 평균']
    
def my_process(x, avg_csv):
    y = interpolate(x, avg_csv)
    return y
    
def reduce_number(x):
    '''
    내부 온습도 1/2/3은 평균을 사용한다.
    '''
    result = pd.Series(0., temper_cols)
    for env in ['온도', '습도']:
        for measure in ['평균','최고','최저']:
            col_list = [f'내부 {env} {i} {measure}' for i in range(1,3+1)]
            y = x[col_list]
            result[f'내부 {env} {measure}'] = np.mean(y[~y.isna()])
    return result
    
def interpolate(x, avg_csv):
    y = x.interpolate(method='polynomial', order=2)
    y = y.fillna(method='ffill').fillna(method='bfill')
    y = y.fillna(avg_csv.loc[x.name, 'mean']) # Fill with averages
    return y

def step_1(txt_path):
    with open(txt_path,'r') as f:
        df_paths = [f"{path.strip()}/{path.split('/')[-1].strip()}.csv" for path in f.readlines()]
    # df_paths = glob('data/sample/*/*.csv')

    statistics = pd.read_csv('dataset/csv_statistics.csv', index_col=0)

    _stats = reduce_number(statistics['mean'])
    _stats.name = 'mean'
    statistics = pd.concat([statistics['mean'], _stats]).to_frame()

    for path in tqdm(df_paths):
        # 이미 존재할 경우 건너뛰기
        if os.path.exists(path.replace('.csv', '_.csv')):
            continue
        
        df = pd.read_csv(path)
        df = df[['측정시각', *use_cols]].replace(['-',0], np.NaN)
        df[use_cols] = df[use_cols].astype(float)
        
        df[temper_cols] = df[use_cols].apply(reduce_number, axis=1)
        df = df[['측정시각', *reduced_cols]]
        
        df[reduced_cols] = df[reduced_cols].apply(lambda x: interpolate(x, statistics), axis=0)
        df.to_csv(path.replace('.csv', '_.csv'), index=False)

def step_2(txt_path):
    with open(txt_path,'r') as f:
        df_paths = [f"{path.strip()}/{path.split('/')[-1].strip()}_.csv" for path in f.readlines()]
    for path in tqdm(df_paths):
        df = pd.read_csv(path)
        if len(df)>2 and (df['측정시각'][0] == df['측정시각'][1] or df['측정시각'][1] == df['측정시각'][2]):
            df = df.loc[::2]
            df.to_csv(path, index=False)
        else:
            continue

def main(args):
    
    txt_path = f"{args.data}/{'Test' if args.test else 'Valid' if args.valid else 'Train'}.txt"
    
    if args.step == 1:
        step_1(txt_path)
    elif args.step == 2:
        step_2(txt_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data', required=True)
    parser.add_argument('-t','--test', action='store_true')
    parser.add_argument('-v','--valid', action='store_true')
    parser.add_argument('-s','--step', default=1, type=int, required=True)
    args = parser.parse_args()
    
    main(args)