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
import tensorflow_addons as tfa
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

'''-----------------read in test image-----------------'''
os.chdir("C://Users//nadja//Documents//Chan Vese Algorithm//Code_sauber//CV+DL")
gt = plt.imread("meningiom.png")
gt =resize(gt,(112,100))
gt1 = rgb2gray(gt[:,:])
gt1 = gt1[6:106,:]
gt1 = np.expand_dims(gt1,axis=0)
img = gt1-np.min(gt1)
img = img/np.max(gt)
img = preprocessing(gt1)

ímg = img-np.min(img)
img = img/np.max(img)


#%%Conditions for the decomposition part
'''-----------Define the norm loss preventig the trivial zero-channel solution--------------------------'''
def Norm_Loss(y_true,y_pred):
    # avoids the norm from each image from being zero or too large (we choose here 1/3 of original image)
    summe=(tf.norm(y_pred[:,:,:,0])-1/4*tf.norm(img))**2  + (tf.norm(y_pred[:,:,:,1])-1/4*tf.norm(img))**2  +(tf.norm(y_pred[:,:,:,2])-1/4*tf.norm(img))**2+(tf.norm(y_pred[:,:,:,3])-1/4*tf.norm(img))**2

    sum_condition = K.sum(tf.ones_like(y_pred[:,:,:,0])- y_pred[:,:,:,0]- y_pred[:,:,:,1]-y_pred[:,:,:,2])
    return summe


'''-----------Define the Coherence penalty term preventing the channels from being overlapping------'''

def coherence_penalty(y_true,y_pred):
    #incoherence \sum log(1-<\Phi_i,Phi_j>/||Phi_i||*||Phi_j||)
    print(y_true.shape)

    coh=-K.sum(tf.math.log(1-(K.sum(y_pred[:,:,:,0:1]*y_pred[:,:,:,1:2])/(tf.norm(y_pred[:,:,:,1])*tf.norm(y_pred[:,:,:,0]))))
               +tf.math.log(1-(K.sum(y_pred[:,:,:,2:3]*y_pred[:,:,:,1:2])/(tf.norm(y_pred[:,:,:,1])*tf.norm(y_pred[:,:,:,2]))))
               +tf.math.log(1-(K.sum(y_pred[:,:,:,0:1]*y_pred[:,:,:,2:3])/(tf.norm(y_pred[:,:,:,0])*tf.norm(y_pred[:,:,:,2]))))
               +tf.math.log(1-(K.sum(y_pred[:,:,:,0:1]*y_pred[:,:,:,2:3])/(tf.norm(y_pred[:,:,:,0])*tf.norm(y_pred[:,:,:,2]))))
               +tf.math.log(1-(K.sum(y_pred[:,:,:,0:1]*y_pred[:,:,:,3:4])/(tf.norm(y_pred[:,:,:,0])*tf.norm(y_pred[:,:,:,3]))))
               +tf.math.log(1-(K.sum(y_pred[:,:,:,1:2]*y_pred[:,:,:,3:4])/(tf.norm(y_pred[:,:,:,1])*tf.norm(y_pred[:,:,:,3]))))
               +tf.math.log(1-(K.sum(y_pred[:,:,:,2:3]*y_pred[:,:,:,3:4])/(tf.norm(y_pred[:,:,:,2])*tf.norm(y_pred[:,:,:,3])))))


    return coh + 0.001*tf.image.total_variation(y_pred[:,:,:,:3])
    


