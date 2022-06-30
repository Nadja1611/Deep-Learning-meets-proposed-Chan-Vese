# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:21:33 2022

@author: nadja
"""

import os
from functions import*
#%%Construction of the decomposition NW used to get the features
def Phi():
    images = Input(shape=(64,64,1), name='images')

    conv1 = Conv2D(32, 3, 1, activation='relu', padding="same")(images)
    conv1 = Conv2D(32, 3, 1, activation='relu',   padding="same")(conv1)
    #print(conv1.shape)

    conv2 = Conv2D(64, 3, 1, activation='relu',padding="same")(conv1)
    conv2 = Conv2D(64, 3, 1, activation='relu', padding='same')(conv2)

    conv3 = Conv2D(128, 3, 1, activation='relu', padding='same')(conv2)
    conv3 = Conv2D(128, 3, 1, activation='relu', padding='same')(conv3)

    conv4 = Conv2D(256, 3, 1, activation='relu', padding='same')(conv3)
    conv4 = Conv2D(256, 3, 1, activation='relu', padding='same')(conv4)



    conv10 = Conv2D(3, 3, 1, activation = 'relu', padding='same')(conv4)
    print(conv10.shape)
    out = conv10
    
    model = Model(inputs=images, outputs=out)
    
    model.compile(optimizer=SGD(lr=1e-3, momentum=0.99, decay=1e-2), loss=[Norm_Loss, coherence_penalty], loss_weights=[0.5,0.5], metrics=['accuracy'])

    return model
model2=Phi()

#%% Reconstruction NW Psi, ground truth here is the original input image

def Psi():
    images = Input(shape=(64,64,3), name='images')

    conv1 = Conv2D(32, 3, 1, activation='relu', padding="same")(images)
    conv1 = Conv2D(32, 3, 1, activation='relu',   padding="same")(conv1)
    #print(conv1.shape)

    conv2 = Conv2D(64, 3, 1, activation='relu',padding="same")(conv1)
    conv2 = Conv2D(64, 3, 1, activation='relu', padding='same')(conv2)

    conv3 = Conv2D(128, 3, 1, activation='relu', padding='same')(conv2)
    conv3 = Conv2D(128, 3, 1, activation='relu', padding='same')(conv3)

    conv4 = Conv2D(256, 3, 1, activation='relu', padding='same')(conv3)
    conv4 = Conv2D(256, 3, 1, activation='relu', padding='same')(conv4)

    conv10 = Conv2D(1, 3, 1, activation = 'sigmoid', padding='same')(conv4)
    print(conv10.shape)
    out = conv10
    
    model = Model(inputs=images, outputs=out)
    
    model.compile(optimizer=SGD(lr=1e-3, momentum=0.99, decay=1e-2), loss="MSE", metrics=['accuracy'])

    return model
model2=Phi()