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
from skimage.color import rgb2gray



#%%preprocessing functions
def preprocessing(X):
    liste=[]
    for i in range(len(X)):
        m = (X[i]-np.mean(X[i]))/np.std(X[i])
        liste.append(m)
    X = np.asarray(liste)
    X=np.expand_dims(X,axis=-1)
    return X



#%%Conditions for the decomposition part
'''-----------Define the norm loss preventig the trivial zero-channel solution--------------------------'''
def Norm_Loss(y_true,y_pred):
    value1 = tf.norm(y_pred[:,:,:,0])
    value2 = tf.norm(y_pred[:,:,:,1])
    value3 = tf.norm(y_pred[:,:,:,2])
    return tf.math.log(value1)+tf.math.log(value2)+tf.math.log(value3)

'''-----------Define the Coherence penalty term preventing the channels from being overlapping------'''
def coherence_penalty(y_true,y_pred):
    coh=-K.sum(tf.math.log(1-tf.multiply(y_pred[:,:,:,1], y_pred[:,:,:,0]))-tf.math.log(1-tf.multiply(y_pred[:,:,:,1], y_pred[:,:,:,2])-tf.math.log(1-tf.multiply(y_pred[:,:,:,2], y_pred[:,:,:,0]))))
    return coh

