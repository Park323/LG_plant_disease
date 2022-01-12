import argparse, yaml
from torch import load
import matplotlib.pyplot as plt

def plot_hist(hist):
   plt.figure(figsize=(40,10))

   plt.subplot(1,2,1)
   plt.plot(hist['train_loss'], label='train')
   plt.plot(hist['val_loss'], label='val')
   plt.title('Loss Graph')
   plt.legend()

   plt.subplot(1,2,2)
   plt.plot(hist['train_f1'], label='train')
   plt.plot(hist['val_f1'], label='val')
   plt.title('F1 Graph')
   plt.legend()

   plt.show()
   
if __name__=='__main__':
    parser = argparse.ArgumentParser
    parser.add_argument('-m','--model',type=str,required=True)
    args = parser.parse_args()
    
    with open(f'config/{args.model}_config.yaml') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    hist = load(f'{config.TRAIN.SAVE_PATH}/train_history.pt')
    
    plot_hist(hist)