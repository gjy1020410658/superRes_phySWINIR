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
parser.add_argument('--batch_size', type = int, default= 64)
parser.add_argument('--crop_size', type = int, default= 128, help= 'should be same as image dimension')
parser.add_argument('--epochs', type = int, default= 100)
parser.add_argument('--seed',type =int, default= 0)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.set_default_dtype(torch.float64)
########### Parameters ############
BATCH_SIZE = args.batch_size # please try 32 it should okay 
NUM_WORKERS = 0 # on Windows, set this variable to 0
scale_factor = args.scale_factor
nb_epochs = args.epochs # typically kernel will crush after 500 epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CROP_SIZE = args.crop_size
gama = args.phy_scale
data_dx = 2*np.pi/2048
########### loaddata ############
trainloader,valloader,_ = getdata(BATCH_SIZE = BATCH_SIZE,NUM_WORKERS =NUM_WORKERS,SEED=args.seed,CROP_SIZE = CROP_SIZE,SCALE_FACTOR = scale_factor)


savedpath = str("model_SwinIR128_P_u." + str(args.scale_factor) + "_Loss_" + str(args.loss_type) + "_FD_kernel_" + str(args.FD_kernel) + "_gama_"
                +str(args.phy_scale) + "_seed_" +str(args.seed)
                )

model = SwinIR(upscale=scale_factor, in_chans=3, img_size=CROP_SIZE, window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_conv='1conv').to(device,dtype=torch.float64)
model = torch.nn.DataParallel(model).to(device)

lossgen = LossGenerator(dx=data_dx,kernel_size= args.FD_kernel).to(device,dtype=torch.float64)

criterion1 = nn.L1Loss().to(device)
criterion2 = nn.MSELoss().to(device)
if args.loss_type =='L1':
    criterion_Data = nn.L1Loss().to(device)
elif args.loss_type =='L2':
    criterion_Data = nn.MSELoss().to(device)

model = SwinIR(upscale=scale_factor, in_chans=3, img_size=CROP_SIZE, window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_conv='1conv').to(device,dtype=torch.float64)
model = torch.nn.DataParallel(model).to(device)

lossgen = LossGenerator(dx=data_dx,kernel_size= args.FD_kernel).to(device,dtype=torch.float64)

criterion1 = nn.L1Loss().to(device)
criterion2 = nn.MSELoss().to(device)

if args.loss_type =='L1':
    criterion_Data = nn.L1Loss().to(device)
elif args.loss_type =='L2':
    criterion_Data = nn.MSELoss().to(device)
    
optimizer = optim.Adam(model.parameters(), lr=0.0001)


hist_loss_train = []
hist_loss_val = []
hist_psnr_train = []
hist_psnr_val = []
hist_phyloss_train = []
hist_phyloss_val = []
hist_dataloss_train = []
hist_dataloss_val = []
best_loss_val = float(999999999)
best_model = model

# resume training in case colab kernal collapse
# checkpoint = torch.load("model_SwinIR.pt")
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


for epoch in range(nb_epochs):

    # Train
    avg_psnr = 0
    avg_phyloss = 0
    epoch_loss = 0
    avg_dataloss = 0
    for iteration, batch in enumerate(tqdm(trainloader)):
        input, target = batch[0].to(device), batch[1].to(device)
        model.train()
        optimizer.zero_grad()
        out = model(input)
        div = lossgen.get_div_loss(output=out)
        phy_loss = criterion2(div,torch.zeros_like(div).to(device)) # DO NOT CHANGE THIS ONE. Phy loss has to be L2 norm 
        data_loss = criterion_Data(out, target) # Experiment change to criterion 1
        loss =  data_loss + gama*phy_loss
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        psnr = 10 * np.log10(1 / loss.item())
        avg_psnr += psnr
        avg_phyloss += phy_loss.item()
        avg_dataloss += data_loss.item()
    print(f"Epoch {epoch}. Training loss: {epoch_loss / len(trainloader)}, Data loss: {avg_dataloss/len(trainloader)}, Phy Loss: {phy_loss.item()}")
    hist_loss_train.append(epoch_loss / len(trainloader))
    hist_psnr_train.append(avg_psnr / len(trainloader))
    hist_phyloss_train.append(avg_phyloss / len(trainloader))
    hist_dataloss_train.append(avg_dataloss / len(trainloader))
    #Test
    avg_psnr = 0
    epoch_loss = 0
    avg_phyloss = 0
    avg_dataloss = 0
    for batch in valloader: # better be the val loader, need to modify datasets, but we are good for now.
        with torch.no_grad():
            input, target = batch[0].to(device), batch[1].to(device)
            model.eval()
            out = model(input)
            div = lossgen.get_div_loss(output=out)
            phy_loss = criterion2(div,torch.zeros_like(div).to(device)) # DO NOT CHANGE THIS ONE. Phy loss has to be L2 norm 
            data_loss = criterion_Data(out, target) # Experiment change to criterion 1
            loss =  data_loss + gama*phy_loss
            psnr = 10 * np.log10(1 / loss.item())
            epoch_loss += loss.item()
            avg_psnr += psnr
            avg_phyloss += phy_loss.item()
            avg_dataloss += data_loss.item()
    print(f"Average PSNR: {avg_psnr / len(valloader)} dB.")
    hist_loss_val.append(epoch_loss / len(valloader))
    hist_psnr_val.append(avg_psnr / len(valloader))
    hist_phyloss_val.append(avg_phyloss / len(valloader))
    hist_dataloss_val.append(avg_dataloss / len(valloader))

    if hist_loss_val[-1] < best_loss_val:
        best_loss_val = hist_loss_val[-1]
        best_model = model

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss / len(valloader),
            'PSNR': avg_psnr / len(valloader)
            },"results/"+savedpath + ".pt" ) # remember to change name for each experiment

#model = best_model
np.save('results/hist_loss_val_' + savedpath,np.array(hist_loss_val))
np.save('results/hist_psnr_val_'+savedpath,np.array(hist_psnr_val))
np.save('results/hist_loss_train_'+ savedpath,np.array(hist_loss_train))
np.save('results/hist_psnr_train_' + savedpath,np.array(hist_psnr_train))
np.save('results/hist_phyloss_train_'+ savedpath,np.array(hist_phyloss_train))
np.save('results/hist_phyloss_val_' + savedpath,np.array(hist_phyloss_val))
np.save('results/hist_dataloss_train_'+ savedpath,np.array(hist_dataloss_train))
np.save('results/hist_dataloss_val_' + savedpath,np.array(hist_dataloss_val))