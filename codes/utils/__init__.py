# global variables
SEED = 42
CLASSES = 5
BATCH_SIZE = 10
INPUT_SHAPE = (512,512,3)
SPLIT_RATIO = 0.2
USE_FLOAT16 = False
DATA_MEAN = [109.73170952, 126.66521077, 79.92212092]
DATA_STD  = [55.86805631, 57.10547278, 51.15093823]

# global imports
import numpy as np
np.random.seed(SEED)
import os
import cv2
from glob import glob
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# use mixed float16 precision on tensorflow
if USE_FLOAT16:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    FLOAT_TYPE = tf.float16
else:
    FLOAT_TYPE = tf.float32

# function to read images in RGB and resize
def imread(path, resize=None):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if resize:
        img = cv2.resize(img, resize[::-1])

    return img

def plot_loss(history, label, n):
    # Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
               color=colors[n], label='Train '+label)
    plt.semilogy(history.epoch, history.history['val_loss'],
          color=colors[n], label='Val '+label,
          linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    
def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()

if __name__ == 'utils':
    from .data_loader import DataLoader
    from .model import Model