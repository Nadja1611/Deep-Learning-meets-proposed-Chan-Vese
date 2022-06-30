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



    conv10a = Conv2D(1, 3, 1, activation = 'relu', padding='same')(conv4)
    conv10b = Conv2D(1, 3, 1, activation = 'relu', padding='same')(conv4)
    conv10c = Conv2D(1, 3, 1, activation = 'relu', padding='same')(conv4)
    conv10 = concatenate([conv10a,conv10b,conv10c],axis=-1)

    print(conv10.shape)
    
    

#%% Reconstruction NW Psi, ground truth here is the original input image

    conv1b = Conv2D(32, 3, 1, activation='relu', padding="same")(conv10a)
    conv1b = Conv2D(32, 3, 1, activation='relu',   padding="same")(conv1b)
    #print(conv1.shape)

    conv2b = Conv2D(64, 3, 1, activation='relu',padding="same")(conv1b)
    conv2b = Conv2D(64, 3, 1, activation='relu', padding='same')(conv2b)

    conv3b = Conv2D(128, 3, 1, activation='relu', padding='same')(conv2b)
    conv3b = Conv2D(128, 3, 1, activation='relu', padding='same')(conv3b)

    conv4b = Conv2D(256, 3, 1, activation='relu', padding='same')(conv3b)
    conv4b = Conv2D(256, 3, 1, activation='relu', padding='same')(conv4b)

    conv10d = Conv2D(1, 3, 1, activation = 'sigmoid', padding='same')(conv4b)
    
    
    
    
    conv1bb = Conv2D(32, 3, 1, activation='relu', padding="same")(conv10b)
    conv1bb = Conv2D(32, 3, 1, activation='relu',   padding="same")(conv1bb)
    #print(conv1.shape)

    conv2bb = Conv2D(64, 3, 1, activation='relu',padding="same")(conv1bb)
    conv2bb = Conv2D(64, 3, 1, activation='relu', padding='same')(conv2bb)

    conv3bb = Conv2D(128, 3, 1, activation='relu', padding='same')(conv2bb)
    conv3bb = Conv2D(128, 3, 1, activation='relu', padding='same')(conv3bb)

    conv4bb = Conv2D(256, 3, 1, activation='relu', padding='same')(conv3bb)
    conv4bb = Conv2D(256, 3, 1, activation='relu', padding='same')(conv4bb)

    conv10dd = Conv2D(1, 3, 1, activation = 'sigmoid', padding='same')(conv4bb)
    
    
    conv1bbb = Conv2D(32, 3, 1, activation='relu', padding="same")(conv10c)
    conv1bbb = Conv2D(32, 3, 1, activation='relu',   padding="same")(conv1bbb)
    #print(conv1.shape)

    conv2bbb = Conv2D(64, 3, 1, activation='relu',padding="same")(conv1bbb)
    conv2bbb = Conv2D(64, 3, 1, activation='relu', padding='same')(conv2bbb)

    conv3bbb = Conv2D(128, 3, 1, activation='relu', padding='same')(conv2bbb)
    conv3bbb = Conv2D(128, 3, 1, activation='relu', padding='same')(conv3bbb)

    conv4bbb = Conv2D(256, 3, 1, activation='relu', padding='same')(conv3bbb)
    conv4bbb = Conv2D(256, 3, 1, activation='relu', padding='same')(conv4bbb)

    conv10ddd = Conv2D(1, 3, 1, activation = 'sigmoid', padding='same')(conv4bbb)
    print(conv10ddd.shape)
    
    recon = tf.keras.layers.add((conv10d,conv10dd,conv10ddd))
  #  out2 = [conv10b,conv10,conv10]
    print(conv10.shape)
    model = Model(inputs=images, outputs=[conv10,conv10,recon])
    
    model.compile(optimizer=SGD(lr=1e-5, momentum=0.99, decay=1e-2), loss=[Norm_Loss,coherence_penalty, "MSE"], metrics=['accuracy'])

    return model
model2=Phi()




model_checkpoint=ModelCheckpoint('Decomposition.hdf5',save_best_only=True, monitor="loss")


h = model2.fit(img,img, epochs=300, batch_size = 1, callbacks=[model_checkpoint], verbose=1)

