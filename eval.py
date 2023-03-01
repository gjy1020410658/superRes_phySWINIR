import argparse
import math
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
from os.path import join
from scipy.io import loadmat
from tqdm import tqdm
import h5py
from src.models import *
from src.utli import *
from src.get_data import getdata

parser = argparse.ArgumentParser(description='training parameters')
parser.add_argument('--loss_type', type =str ,default= 'L1')
parser.add_argument('--phy_scale', type = float, default= 0.5,help= 'physics loss factor')
parser.add_argument('--FD_kernel', type = int, default= 3) # or 5
parser.add_argument('--scale_factor', type = int, default= 8)
parser.add_argument('--batch_size', type = int, default= 32)
parser.add_argument('--crop_size', type = int, default= 128, help= 'should be same as image dimension')
parser.add_argument('--epochs', type = int, default= 100)
parser.add_argument('--seed',type =int, default= 0)
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

torch.set_default_dtype(torch.float64)

BATCH_SIZE = args.batch_size # please try 32 it should okay 
NUM_WORKERS = 0 # on Windows, set this variable to 0
scale_factor = args.scale_factor
nb_epochs = args.epochs # typically kernel will crush after 500 epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CROP_SIZE = args.crop_size
gama = args.phy_scale
########### loaddata ############
_,_,testloader = getdata(BATCH_SIZE = BATCH_SIZE,NUM_WORKERS =NUM_WORKERS,SEED=args.seed,CROP_SIZE = CROP_SIZE,SCALE_FACTOR = scale_factor)

data_dx = 2*np.pi/2048

########### loaddata ############

savedpath = str("model_SwinIR128_P_u." + str(args.scale_factor) + "_Loss_" + str(args.loss_type) + "_FD_kernel_" + str(args.FD_kernel) + "_gama_"
                +str(args.phy_scale) + "_seed_" +str(args.seed)
                )

checkpoint = torch.load("results/" + savedpath + ".pt")
model = SwinIR(upscale=scale_factor, in_chans=3, img_size=CROP_SIZE, window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_conv='1conv').to(device,dtype=torch.float64)
model = torch.nn.DataParallel(model).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
lossgen = LossGenerator(dx=data_dx,kernel_size= args.FD_kernel).to(device,dtype=torch.float64)

epoch_model = checkpoint['epoch']
loss_model = checkpoint['loss']

model.eval()
avg_psnr = 0
epoch_loss = 0
avg_phyloss = 0
MSEfunc = nn.MSELoss()
errors_L1 = []
error_L2 = 0
top_err = 0
bot_err = 0

with torch.no_grad():
    for batch in testloader:
        input, target = batch[0].to(device), batch[1].to(device)
        model.eval()
        out = model(input)
        div = lossgen.get_div_loss(output=out)
        phy_loss = MSEfunc(div,torch.zeros_like(div).to(device))
        loss = MSEfunc(out, target)
        top_err += MSEfunc(out, target).item()
        bot_err += MSEfunc(target,torch.zeros_like(target)).item()
        psnr = 10 * np.log10(1 / loss.item())
        epoch_loss += loss.item()
        avg_psnr += psnr
        avg_phyloss += phy_loss.item()

with open('results/testresult.txt',"a") as ff:
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",file=ff)
    print(args,file=ff)
    # print('model at: %.1f, loss: %.6f' % epoch_model, loss_model,file=ff)
    print(f"Average PSNR: {avg_psnr / len(testloader)} dB.",file = ff)
    print('relative error: %.6f' % np.sqrt(top_err/bot_err),file = ff)
    print(f"aRMSE: { epoch_loss/len(testloader)} ",file = ff)
    print(f"divergence loss: { avg_phyloss/len(testloader)} ",file = ff)

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",file=ff)
    ff.close()