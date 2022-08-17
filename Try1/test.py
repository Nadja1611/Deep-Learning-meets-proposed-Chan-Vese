# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 20:20:46 2022

@author: johan
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean
from torch.utils.data import TensorDataset, DataLoader


im_size = 192

class encoder(torch.nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()  
        # define layers
        self.conv1 = nn.Conv2d(in_channels = input_channels, out_channels = 16 , kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 16 , kernel_size = 3, padding = 1)
        self.down1 = nn.MaxPool2d(kernel_size = 2)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 16 , kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 16 , kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 16, out_channels = 16 , kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(in_channels = 16, out_channels = 16 , kernel_size = 3, padding = 1)
        self.conv7 = nn.Conv2d(in_channels = 16, out_channels = 16 , kernel_size = 3, padding = 1)
        self.conv8 = nn.Conv2d(in_channels = 16, out_channels = 16 , kernel_size = 3, padding = 1)
        self.lin1a = nn.Linear(in_features = 16*576,out_features = 6)
        self.lin1b = nn.Linear(in_features = 16*576,out_features = 6)
        self.lin2a = nn.Linear(in_features = 16*576,out_features = 6)
        self.lin2b = nn.Linear(in_features = 16*576,out_features = 6)
        self.lin3a = nn.Linear(in_features = 16*576,out_features = 6)
        self.lin3b = nn.Linear(in_features = 16*576,out_features = 6)
        self.lin4a = nn.Linear(in_features = 16*576,out_features = 6)
        self.lin4b = nn.Linear(in_features = 16*576,out_features = 6)
        self.actr = nn.ReLU()

    def forward(self, x):
        x = self.actr(self.conv1(x))
        x = self.actr(self.conv2(x))
        x = self.down1(x)
        x = self.actr(self.conv3(x))
        x = self.actr(self.conv4(x))
        x = self.down1(x)
        x = self.actr(self.conv5(x))
        x = self.actr(self.conv6(x))#
        x = self.down1(x)
        x = self.actr(self.conv7(x))
        x = self.actr(self.conv8(x))
        x = x.flatten(start_dim = 1)
        mu1 = self.lin1a(x)
        sig1 = self.lin1b(x)
        mu2 = self.lin2a(x)
        sig2 = self.lin2b(x)
        mu3 = self.lin3a(x)
        sig3 = self.lin3b(x)
        mu4 = self.lin4a(x)
        sig4 = self.lin4b(x)
              
        return mu1,sig1,mu2,sig2,mu3,sig3, mu4, sig4
    
    
class decoder(torch.nn.Module):

    def __init__(self, input_channels, output_channels,im_size):
        super().__init__()  
        # define layers
        self.im_size = im_size
        self.lin1a = nn.Linear(in_features = 6, out_features = 128)
        self.lin2a = nn.Linear(in_features = 6, out_features = 128)
        self.lin3a = nn.Linear(in_features = 6, out_features = 128)
        self.lin4a = nn.Linear(in_features = 6, out_features = 128)
        self.lin1 = nn.Linear(in_features = 128, out_features = 4*(im_size//2)**2)
        self.lin2 = nn.Linear(in_features = 128, out_features = 4*(im_size//2)**2)
        self.lin3 = nn.Linear(in_features = 128, out_features = 4*(im_size//2)**2)
        self.lin4 = nn.Linear(in_features = 128, out_features = 4*(im_size//2)**2)
        self.up1 = nn.UpsamplingBilinear2d(size = [im_size,im_size])
        self.conv1 = nn.Conv2d(in_channels = 4, out_channels = 32 , kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 32 , kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 4, out_channels = 32 , kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 4, out_channels = 32 , kernel_size = 3, padding = 1)
        self.conv1b = nn.Conv2d(in_channels = 32, out_channels = 1 , kernel_size = 3, padding = 1)
        self.conv2b = nn.Conv2d(in_channels = 32, out_channels = 1 , kernel_size = 3, padding = 1)
        self.conv3b = nn.Conv2d(in_channels = 32, out_channels = 1 , kernel_size = 3, padding = 1)
        self.conv4b = nn.Conv2d(in_channels = 32, out_channels = 1 , kernel_size = 3, padding = 1)

        self.actr = nn.ReLU()

    def forward(self, z1,z2,z3,z4,U):
        x1 = self.actr(self.lin1a(z1))
        x2 = self.actr(self.lin2a(z2))
        x3 = self.actr(self.lin3a(z3))
        x4 = self.actr(self.lin4a(z4))
        x1 = self.actr(self.lin1(x1))
        x2 = self.actr(self.lin2(x2))
        x3 = self.actr(self.lin3(x3))
        x4 = self.actr(self.lin4(x4))
        x1 = torch.reshape(x1,(4,4,self.im_size//2,self.im_size//2))
        x2 = torch.reshape(x2,(4,4,self.im_size//2,self.im_size//2))
        x3 = torch.reshape(x3,(4,4,self.im_size//2,self.im_size//2))
        x4 = torch.reshape(x4,(4,4,self.im_size//2,self.im_size//2))
        x1 = self.up1(x1)
        x2 = self.up1(x2)
        x3 = self.up1(x3)
        x4 = self.up1(x4)
        x1 = self.actr(self.conv1(x1))
        x2 = self.actr(self.conv2(x2))
        x3 = self.actr(self.conv3(x3))
        x4 = self.actr(self.conv4(x4))
        x1 = self.conv1b(x1)
        x2 = self.conv2b(x2)
        x3 = self.conv3b(x3)
        x4 = self.conv4b(x4) 
        print(x1.shape)
        x = torch.sum(torch.cat([x1,x2,x3,x4],1),dim = 1).unsqueeze(0)
        print(x.shape)
        return x, x1, x2, x3, x4

def norm_loss(y_true,y_pred):
    im_norm = torch.linalg.norm(y_true)
    summe=(torch.linalg.norm(y_pred[:,0,:,:])-1/4*im_norm)**2  + (torch.linalg.norm(y_pred[:,1,:,:])-1/4*im_norm)**2 + (torch.linalg.norm(y_pred[:,2,:,:])-1/4*im_norm)**2 +(torch.linalg.norm(y_pred[:,3,:,:])-1/4*im_norm)**2
    return summe


def coherence_penalty(y_pred):

    coh=-(torch.log(1-(torch.sum(y_pred[:,0:1,:,:]*y_pred[:,1:2,:,:])/(torch.linalg.norm(y_pred[:,1:2,:,:])*torch.linalg.norm(y_pred[:,0:1,:,:])+1e-8))**2) \
               +torch.log(1-(torch.sum(y_pred[:,2:3,:,:]*y_pred[:,1:2,:,:])/(torch.linalg.norm(y_pred[:,1:2,:,:])*torch.linalg.norm(y_pred[:,2:3,:,:])+1e-8))**2) \
               +torch.log(1-(torch.sum(y_pred[:,0:1,:,:]*y_pred[:,2:3,:,:])/(torch.linalg.norm(y_pred[:,0:1,:,:])*torch.linalg.norm(y_pred[:,2:3,:,:])+1e-8))**2) \
               +torch.log(1-(torch.sum(y_pred[:,0:1,:,:]*y_pred[:,3:4,:,:])/(torch.linalg.norm(y_pred[:,0:1,:,:])*torch.linalg.norm(y_pred[:,3:4,:,:])+1e-8))**2) \
               +torch.log(1-(torch.sum(y_pred[:,1:2,:,:]*y_pred[:,3:4,:,:])/(torch.linalg.norm(y_pred[:,1:2,:,:])*torch.linalg.norm(y_pred[:,3:4,:,:])+1e-8))**2) \
               +torch.log(1-(torch.sum(y_pred[:,2:3,:,:]*y_pred[:,3:4,:,:])/(torch.linalg.norm(y_pred[:,2:3,:,:])*torch.linalg.norm(y_pred[:,3:4,:,:])+1e-8))**2))


    return coh 
X=[]
device = 'cuda:0'
for i in range(1,11):
    in_data = np.load("D:\Test_ISLES\Test_ISLES_"+str(i)+".npz", allow_pickle = True)
    dwi=in_data["X"]
    L=in_data["L"]
    X.append(dwi)
X = np.asarray(np.concatenate(X,axis=0))    
X = torch.tensor(X).float().squeeze()
    
input_im =  plt.imread("menigiom1.png") 
input_im = rgb2gray(input_im[:,:,:3])  
#input_im = rescale(input_im,0.1)
input_im = torch.tensor(input_im).unsqueeze(0).unsqueeze(0).to(device)

dataset = TensorDataset(X)
dataloader = DataLoader(dataset, batch_size = 4, shuffle = True,drop_last = True)

im_size = X.shape[-1]
enc = encoder(1,4).to(device)
dec = decoder(1,1,im_size).to(device)

enc_params = enc.parameters()
dec_params = dec.parameters()

LR =1e-4
U = torch.nn.Parameter(torch.stack(4*[torch.rand_like(X)],3),requires_grad = True)

enc_optimizer = torch.optim.Adam(enc_params, lr=LR)
dec_optimizer = torch.optim.Adam(dec_params, lr=LR)
U_optimizer = torch.optim.Adam([U],lr = LR)

N_epochs = 2000
loss_curve = torch.zeros(N_epochs)

for epoch in range(N_epochs):
    for x_in in enumerate(dataloader):
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        x_in = x_in[1][0]
        mu1,sig1,mu2,sig2,mu3,sig3,mu4,sig4 = enc(x_in.unsqueeze(1).to(device))
        z1 = mu1+torch.exp(sig1*0.5)*torch.randn_like(mu1)
        z2 = mu2+torch.exp(sig2*0.5)*torch.randn_like(mu2)
        z3 = mu3+torch.exp(sig3*0.5)*torch.randn_like(mu3)
        z4 = mu4+torch.exp(sig4*0.5)*torch.randn_like(mu4)
        y, c1, c2, c3, c4 = dec(z1,z2,z3,z4,U)
        kld_loss1 = -0.5*torch.mean(torch.sum(1+sig1-mu1**2-torch.exp(sig1),dim=1),dim=0)
        kld_loss2 = -0.5*torch.mean(torch.sum(1+sig2-mu2**2-torch.exp(sig2),dim=1),dim=0)
        kld_loss3 = -0.5*torch.mean(torch.sum(1+sig3-mu3**2-torch.exp(sig3),dim=1),dim=0)
        kld_loss4 = -0.5*torch.mean(torch.sum(1+sig4-mu4**2-torch.exp(sig4),dim=1),dim=0)

        loss = torch.mean((x_in.to(device) -y.squeeze())**2)+0.01*(kld_loss1+kld_loss2+kld_loss3+kld_loss4)+0.5*coherence_penalty(torch.cat([c1,c2,c3,c4],1))
        loss.backward()
    
        enc_optimizer.step()
        dec_optimizer.step()
    
        loss_curve[epoch] = loss.item()
        print(loss.item())
        if epoch%10==0:
               
            f = plt.figure(figsize=(8,4))    
            plt.subplot(231)
            plt.imshow(y[0,0].detach().cpu(),cmap = 'inferno')
            plt.subplot(232)
            plt.imshow(x_in[0],cmap = 'inferno')
            plt.subplot(233)
            plt.imshow(c1[0,0].detach().cpu(),cmap = 'inferno')
            plt.subplot(234)
            plt.imshow(c2[0,0].detach().cpu(),cmap = 'inferno')
            plt.subplot(235)
            plt.imshow(c3[0,0].detach().cpu(),cmap = 'inferno')
            plt.subplot(236)
            plt.imshow(c4[0,0].detach().cpu(),cmap = 'inferno')
    
            plt.show()

    
    plt.plot(loss_curve)