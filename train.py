# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:49:48 2022

@author: nadja
"""

import os 
from functions import*
from models import *


'''read in test image---'''



import os


gt = plt.imread("menigiom1.jpg")
gt =resize(gt,(112,100))
gt1 = rgb2gray(gt[:,:])
gt1 = gt1[6:106,:]
gt1 = np.expand_dims(gt1,axis=0)
img = preprocessing(gt1)


h = model2.fit(img,img, epochs=300, batch_size = 1, callbacks=[model_checkpoint], verbose=1)

