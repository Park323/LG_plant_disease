import yaml
import os
import numpy as np

if __name__=='__main__':
    with open('config/config.yaml') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    data_dir = config['DATA']['DATA_ROOT']
    img_folder = f"{config['DATA']['DATA_ROOT']}/{config['DATA']['IMAGE_PATH']}"
    test_folder = f"{config['DATA']['DATA_ROOT']}/{config['DATA']['TEST_FOLDER']}"
    val_rate = config['DATA']['VALID_RATE']
    train_path = config['DATA']['TRAIN_PATH']
    valid_path = config['DATA']['VALID_PATH']
    test_path  = config['DATA']['TEST_PATH']
    
    JOIN = lambda x: img_folder + '/' + x
    files = os.listdir(img_folder)
    files = [file for file in list(map(JOIN, files)) if os.path.isdir(file)]
    np.random.shuffle(files)
    
    train_files = files[:-int(len(files) * val_rate)]
    valid_files = files[-int(len(files) * val_rate):]
    
    JOIN = lambda x: test_folder + '/' + x
    test_files = os.listdir(test_folder)
    test_files = [file for file in list(map(JOIN, test_files)) if os.path.isdir(file)]
    
    JOIN = lambda x: data_dir + '/' + x
    with open(JOIN(train_path), 'w') as f:
        f.write('\n'.join(train_files))
    with open(JOIN(valid_path), 'w') as f:
        f.write('\n'.join(valid_files))
    with open(JOIN(test_path), 'w') as f:
        f.write('\n'.join(test_files))