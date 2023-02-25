import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import savemat
from scipy.io import loadmat
import torch.nn.init as init
from math import log10
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from os import listdir
from os.path import join
from scipy.io import loadmat
from tqdm import tqdm
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