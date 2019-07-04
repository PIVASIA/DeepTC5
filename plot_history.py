import os
import argparse
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

parser     = argparse.ArgumentParser()
parser.add_argument('--log_dir', default=".", help='Path to dir containing training log')
parser.add_argument('--csv_train', default="train.csv", help='Name of training log')
parser.add_argument('--csv_val', default="eval.csv", help='Name of validation log')
parser.add_argument('--title', help='Plot title', type=str, default="")
parser = parser.parse_args()

def plotloss(log_dir, csv_train, csv_val):
    '''
    Args
        csv_train: name of the csv file
        csv_val: name of the csv file
    Returns
        graph_loss: trend of loss values over epoch
    '''
    # Bring in the csv file
    training_hist   = pd.read_csv(os.path.join(log_dir, csv_train))
    training_hist.drop_duplicates(subset="epoch", keep='last', inplace=True)

    val_hist        = pd.read_csv(os.path.join(log_dir, csv_val))
    val_hist.drop_duplicates(subset="epoch", keep='last', inplace=True)

    hist            = pd.merge(training_hist, val_hist, on='epoch')

    # Initiation
    epoch           = hist["epoch"]
    tr_loss         = hist["loss"]
    val_acc         = np.asarray(hist["mAP"])

    epoch           = epoch[1:] + 1
    tr_loss         = tr_loss[1:]
    val_acc         = val_acc[1:]

    fig, ax1        = plt.subplots(figsize=(8, 6))
    ax2             = ax1.twinx()

    # Label and color the axes
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Training loss', fontsize=16, color='black')
    ax2.set_ylabel('Validation mAP', fontsize=16, color='black')

    # Plot valid/train losses
    ax1.plot(epoch, tr_loss, linewidth=2,
             ls='--', color='#c92508', label='Train loss')
    ax1.spines['left'].set_color('#f23d1d')
    # Coloring the ticks
    for label in ax1.get_yticklabels():
        label.set_color('#c92508')
        label.set_size(12)

    # Plot valid/trian accuracy
    ax2.plot(epoch, val_acc, linewidth=2,
             color='#2348ff', label='Validation mAP')
    ax2.spines['right'].set_color('#2348ff')
    # Coloring the ticks
    for label in ax2.get_yticklabels():
        label.set_color('#2348ff')
        label.set_size(12)

    # Manually setting the y-axis ticks
    yticks = np.arange(0, 1.1, 0.1)
    ax2.set_yticks(yticks)

    for label in ax1.get_xticklabels():
        label.set_size(12)

    # Modification of the overall graph
    fig.legend(ncol=4, loc=9, fontsize=12)
    plt.xlim(xmin=0)
    ax2.set_ylim(ymax=1, ymin=0)
    plt.title(parser.title, weight="bold")
    plt.grid(True, axis='y')

if __name__ == '__main__':
    plt.show(plotloss(parser.log_dir, parser.csv_train, parser.csv_val))
