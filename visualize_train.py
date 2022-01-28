import argparse
import torch
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-h','--hist_path',
                        required=True, type=str)
    args = parser.parse_args()
    
    hist = torch.load(args.hist_path)
    
    plot_hist(hist)

