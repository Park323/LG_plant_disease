import yaml
import os
import numpy as np

if __name__=='__main__':
    with open('config/config.yaml') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    data_dir = config['DATA']['DATA_PATH']
    val_rate = config['DATA']['VALID_RATE']
    train_path = config['DATA']['TRAIN_PATH']
    valid_path = config['DATA']['VALID_PATH']
    
    JOIN = lambda x: data_dir + '/' + x
    files = os.listdir(data_dir)
    files = [file for file in list(map(JOIN, files)) if os.path.isdir(file)]
    np.random.shuffle(files)
    
    train_files = files[:-int(len(files) * val_rate)]
    valid_files = files[-int(len(files) * val_rate):]
    
    with open(JOIN(train_path), 'w') as f:
        f.write(','.join(train_files))
    with open(JOIN(valid_path), 'w') as f:
        f.write(','.join(valid_files))