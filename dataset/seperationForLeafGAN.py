import argparse, os, shutil
import json
from glob import glob

'''
Example:
    python dataset/seperationForLeafGAN.py -a 1_00_0 -b 3_00_0 -d data/sample -t data/GAN 
'''

def main(args):
    save_path = args.to_folder
    if not os.path.exists(f'{save_path}/trainA'):
        os.makedirs(f'{save_path}/trainA')
    if not os.path.exists(f'{save_path}/trainB'):
        os.makedirs(f'{save_path}/trainB')
    json_list = glob(f'{args.data_folder}/*/*.json')
    for js in json_list:
        with open(js, 'r') as f:
            annots = json.load(f)
        crop = annots['annotations']['crop']
        disease = annots['annotations']['disease']
        risk = annots['annotations']['risk']
        label = f'{crop}_{disease}_{risk}'
        print(label)
        if label == args.labelA:
            shutil.copy(f"{js.replace('.json','.jpg')}", f'{save_path}/trainA')
        if label == args.labelB:
            shutil.copy(f"{js.replace('.json','.jpg')}", f'{save_path}/trainB')
            
if __name__=='__main__':
    
    # Load Config
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d','--data_folder', default='none')
    parser.add_argument('-t','--to_folder', default='none')
    parser.add_argument('-a','--labelA', default='none')
    parser.add_argument('-b','--labelB', default='none')
    
    args = parser.parse_args()
    
    main(args)