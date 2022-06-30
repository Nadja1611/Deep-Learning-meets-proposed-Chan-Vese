# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:21:33 2022

@author: nadja
"""

import os
from functions import*
#%%Construction of the decomposition NW used to get the features
def Phi():
    images = Input(shape=(100,100,1), name='images')

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
    
    

#%% Reconstruction NW Psi, ground truth here is the original input image

    conv1b = Conv2D(32, 3, 1, activation='relu', padding="same")(conv10)
    conv1b = Conv2D(32, 3, 1, activation='relu',   padding="same")(conv1b)
    #print(conv1.shape)

    conv2b = Conv2D(64, 3, 1, activation='relu',padding="same")(conv1b)
    conv2b = Conv2D(64, 3, 1, activation='relu', padding='same')(conv2b)

    conv3b = Conv2D(128, 3, 1, activation='relu', padding='same')(conv2b)
    conv3b = Conv2D(128, 3, 1, activation='relu', padding='same')(conv3b)

    conv4b = Conv2D(256, 3, 1, activation='relu', padding='same')(conv3b)
    conv4b = Conv2D(256, 3, 1, activation='relu', padding='same')(conv4b)

    conv10b = Conv2D(1, 3, 1, activation = 'sigmoid', padding='same')(conv4b)
    print(conv10b.shape)
  #  out2 = [conv10b,conv10,conv10]
    print(conv10.shape)
    out2 = [conv10b, conv10]
    print(len(out2))
    model = Model(inputs=images, outputs=[conv10,conv10,conv10b])
    
    model.compile(optimizer=SGD(lr=1e-3, momentum=0.99, decay=1e-2), loss=[Norm_Loss,coherence_penalty, "MSE"], metrics=['accuracy'])

    return model
model2=Phi()




model_checkpoint=ModelCheckpoint('Decomposition.hdf5',save_best_only=True, monitor="loss")


h = model2.fit(img,img, epochs=300, batch_size = 1, callbacks=[model_checkpoint], verbose=1)

