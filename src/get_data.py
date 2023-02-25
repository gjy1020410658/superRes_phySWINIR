import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import matplotlib.pyplot as plt


import torch.nn.init as init
from math import log10
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from os import listdir
import h5py
class DatasetFromTensor(Dataset):
    def __init__(self, data, scale_factor, with_bicubic_upsampling = True, CROP_SIZE = 256):
        super(DatasetFromTensor, self).__init__()
        self.data = data

        crop_size = CROP_SIZE - (CROP_SIZE % scale_factor) # Valid crop size
        
        if with_bicubic_upsampling:
            self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size), # cropping the image
                                        transforms.Resize(crop_size//scale_factor),  # subsampling the image (half size)
                                        transforms.Resize(crop_size, interpolation=Image.BICUBIC)  # bicubic upsampling to get back the original size 
                                        ])
        else:
            self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size), # cropping the image
                                        transforms.Resize(crop_size//scale_factor)  # subsampling the image (half size)
                                        ])
                
        self.target_transform = transforms.Compose([transforms.CenterCrop(crop_size) # since it's the target, we keep its original quality
                                        ])

    def __getitem__(self, index):
        input = self.data[index]
        target = input.clone()

        GB = transforms.GaussianBlur(kernel_size=(3,3), sigma=(1,1))
        
        input = GB(input)
        input = self.input_transform(input)
        target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.data)
    
def getdata(name = 'NSKT',BATCH_SIZE = 6,NUM_WORKERS =0,SEED=0,CROP_SIZE = 128,SCALE_FACTOR = 8):
    ########### loaddata ############
    f = h5py.File("../capstone_184/nskt[256,uvw].h5", mode="r")
    w = f["w"][()]
    u = f["u"][()]
    v = f["v"][()]
    # in h5 file channel is for different sub-region

    print(w.shape) # (1800//3=600,2,256,256)
    print(u.shape)

    data_w = w[:,:,64:192,64:192]
    data_u = u[:,:,64:192,64:192]
    data_v = v[:,:,64:192,64:192]
    train_w = np.zeros((data_w.shape[0]*data_w.shape[1],1,data_w.shape[2],data_w.shape[3]))
    train_u = np.zeros((data_w.shape[0]*data_w.shape[1],1,data_w.shape[2],data_w.shape[3]))
    train_v = np.zeros((data_w.shape[0]*data_w.shape[1],1,data_w.shape[2],data_w.shape[3]))
    for i in range (data_w.shape[0]):
        for j in range (data_w.shape[1]):
            train_w[i+j,0,:,:] = data_w[i,j,...]
            train_u[i+j,0,:,:] = data_u[i,j,...]
            train_v[i+j,0,:,:] = data_v[i,j,...]
    dataall = np.concatenate((train_w,train_u,train_v),axis =1)
    # train : test : val = 8:1:1

    dataset = DatasetFromTensor(torch.tensor(dataall,dtype= torch.float64), CROP_SIZE = CROP_SIZE,scale_factor=SCALE_FACTOR, with_bicubic_upsampling=False)

    trianset,testValset = random_split(dataset,[0.8,0.2],generator=torch.Generator().manual_seed(SEED))
    valset,testset = random_split(testValset,[0.5,0.5],generator=torch.Generator().manual_seed(SEED))

    trainloader = DataLoader(dataset=trianset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valloader =  DataLoader(dataset=valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    return trainloader,valloader,testloader