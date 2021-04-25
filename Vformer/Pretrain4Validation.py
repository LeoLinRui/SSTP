# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import argparse
import os
import sys
import numpy as np
import math
import time
import pickle
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import random
from cv2 import VideoWriter, VideoWriter_fourcc, imread

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

import warnings

from ModelZoo import TemporalDiscriminator, StyleGanGenerator, StyleGanDiscriminator

BATCH_SIZE = 2


class Dataset(Dataset):

    def __init__(self, file_dir, transform=None):

        self.dir = file_dir
        self.transform = transform
        self.diction = {}
        
        idx = 0
        for filename in os.listdir(self.dir):
            if filename.endswith('jpg'):
                self.diction[idx] = filename
                idx += 1
                        
    def __len__(self):
        return len(self.diction)

    
    def __getitem__(self, idx):
        x = self.diction[idx]
        directory_x = self.dir + "\\" + str(x)
        x = cv.imread(directory_x) / 255
        if self.transform:
            x = self.transform(x)
        x = torch.Tensor(x)
        x = F.interpolate(x.unsqueeze(0), size=(128, 128)).squeeze(0)
        return x

    
def HWC2CHW(x):
    return np.array(x).transpose(2, 0, 1)


# %%
dataset = Dataset(file_dir=r"C:\Users\Leo's PC\Documents\SSTP Tests\SSTP\Vformer\test_frames", transform=HWC2CHW)
loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)


# %%
from vformer_model import Conv, CombinedEmbedding, DeConv
test_model = nn.Sequential(Conv(dic_size=8192),
                           CombinedEmbedding(vocab_size=8192, positional_encoding=False, device='cuda'),
                           DeConv(input_dim=512))


# %%
# Loss function
MSE_loss = nn.L1Loss()
L1_loss = nn.L1Loss().cuda()


# Initialize generator and discriminator
model = test_model.cuda()
# D = StyleGanDiscriminator()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = torch.nn.DataParallel(model)
# D = torch.nn.DataParallel(D)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))

scaler = GradScaler()

training_log = {'iteration':[], 'loss':[]}


# %%
warnings.filterwarnings("ignore", category=UserWarning)

for epoch in range(300):
    
    start_time = time.time()
    total_loss = 0
    
    for i, imgs in enumerate(loader):

        imgs = Variable(imgs.half()).cuda()

        optimizer.zero_grad()

        with autocast():
            out = model(imgs)

        loss = MSE_loss(out, imgs)
        scaler.scale(loss).backward()
        scaler.step(optimizer) 
        scaler.update()
        
        total_loss += loss
        
    print('Epoch {:d} | {:.2f} minutes | loss: {:.6f}'.format(epoch, (time.time() - start_time) / 60, total_loss/len(loader)*BATCH_SIZE))
    
    torchvision.utils.save_image(out[0], os.path.join(r"C:/Users/Leo's PC/Documents/SSTP Tests/SSTP/Vformer/out_samples", str(epoch) + '.jpg'))
    
    if epoch%10 == 0:
        with open(r"C:/Users/Leo's PC/Documents/SSTP Tests/SSTP/Vformer/model_checkpoints/" + str(epoch), 'wb') as checkpoint_file:
            torch.save({'model': model.state_dict()}, checkpoint_file)
        
    training_log['iteration'].append(epoch)
    training_log['loss'].append(total_loss/len(loader))


# %%



# %%



