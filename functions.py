# -*- coding: utf-8 -*-
"""
Deep Learning and Chan Vese Fusion-functions

@author: nadja
"""
import os
import tensorflow as tf
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D,Dropout, Input, concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers
from scipy import ndimage, misc

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
import matplotlib.pylab as plt
from skimage.transform import rescale, resize, downscale_local_mean



#%%Conditions for the decomposition part
'''-----------Define the norm loss preventig the trivial zero-channel solution--------------------------'''
def Norm_Loss(y_pred,y_true):
    value1 = tf.norm(y_pred[:,:,:,0])
    value2 = tf.norm(y_pred[:,:,:,1])
    value3 = tf.norm(y_pred[:,:,:,2])
    return 1-(value1+value2+value3)

'''-----------Define the Coherence penalty term preventing the channels from being overlapping------'''
def coherence_penalty(y_pred, true):
    coh=1-tf.multiply(y_pred[:,:,:,1], y_pred[:,:,:,0])-tf.multiply(y_pred[:,:,:,1], y_pred[:,:,:,2])-tf.multiply(y_pred[:,:,:,2], y_pred[:,:,:,0])
    return coh

