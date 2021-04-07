import argparse
import os
import numpy as np
import math
import pickle
import cv2 as cv
import matplotlib.pyplot as plt

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


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        #expected inout size (N, 3, 256, 128)
        
        #=======
        #Encoder
        #=======
        
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[3,3], stride=1, padding=1)
        self.activation1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3,3], stride=1, padding=1)
        self.activation1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = [2, 2], stride=2, padding=0)
        
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3,3], stride=1, padding=1)
        self.activation2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3,3], stride=1, padding=1)
        self.activation2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = [2, 2], stride=2, padding=0)
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3,3], stride=1, padding=1)
        self.activation3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=2, padding=1)
        self.activation3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=1, padding=1)
        self.activation3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = [2, 2], stride=2, padding=0)
        
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3,3], stride=1, padding=1)
        self.activation4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=2, padding=1)
        self.activation4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=1, padding=1)
        self.activation4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size = [2, 2], stride=2, padding=0)
        
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=1, padding=1)
        self.activation5_1 = nn.ReLU()      
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=2, padding=1)
        self.activation5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=1, padding=1)
        self.activation5_3 = nn.ReLU()
        
        #size = (N, 512, 4, 2)
        
        #=======
        #Recurrent Module
        #=======
        
        self.pool_r = nn.AdaptiveMaxPool2d(output_size = (2, 1))
        
        #size of GRU input = (batch_size, seq_len, inp_size)
        self.gru = nn.GRU(input_size=1024, hidden_size=1024, num_layers=2, bias=True, batch_first=True, dropout=0.2)
        
        self.upsample_r = nn.Upsample(size=(8, 4))
        
        #=======
        #Decoder
        #=======
        
        #size = (N, 512, 8, 4)
        
        self.conv6_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=1, padding=1)
        self.activation6_1 = nn.ReLU()
        self.conv6_2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=1, padding=1)
        self.activation6_2 = nn.ReLU()
        self.conv6_3 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=1, padding=1)
        self.activation6_3 = nn.ReLU()
        self.upsample6 = nn.Upsample(scale_factor=2)
        
        self.conv7_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=1, padding=1)
        self.activation7_1 = nn.ReLU()
        self.conv7_2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=1, padding=1)
        self.activation7_2 = nn.ReLU()
        self.conv7_3 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=[3,3], stride=1, padding=1)
        self.activation7_3 = nn.ReLU()
        self.upsample7 = nn.Upsample(scale_factor=2)
        
        self.conv8_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=[3,3], stride=1, padding=1)
        self.activation8_1 = nn.ReLU()
        self.conv8_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=1, padding=1)
        self.activation8_2 = nn.ReLU()
        self.conv8_3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=[3,3], stride=1, padding=1)
        self.activation8_3 = nn.ReLU()
        self.upsample8 = nn.Upsample(scale_factor=2)
        
        self.conv9_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=[3,3], stride=1, padding=1)
        self.activation9_1 = nn.ReLU()
        self.conv9_2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=[3,3], stride=1, padding=1)
        self.activation9_2 = nn.ReLU()
        self.upsample9 = nn.Upsample(scale_factor=2)
        
        self.conv10_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=[3,3], stride=1, padding=1)
        self.activation10_1 = nn.ReLU()
        self.conv10_2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=[3,3], stride=1, padding=1)
        self.activation10_2 = nn.ReLU()
        
        self.output = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=[1,1], stride=1, padding=0)
        self.activation_output = nn.Sigmoid()
        
        self.h = torch.Tensor(np.random.randn(2, 2, 1024)) #(num_layers, batchsize, inp.shape)
        

    def forward(self, x, h):
        
        #=======
        #Encoder
        #=======
        
        x = self.conv1_1(x)
        x = self.activation1_1(x)
        x = self.conv1_2(x)
        x = self.activation1_2(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.activation2_1(x)
        x = self.conv2_2(x)
        x = self.activation2_2(x)
        x = self.pool2(x)
        
        x = self.conv3_1(x)
        x = self.activation3_1(x)
        x = self.conv3_2(x)
        x = self.activation3_2(x)
        x = self.conv3_3(x)
        x = self.activation3_3(x)
        x = self.pool3(x)
        
        x = self.conv4_1(x)
        x = self.activation4_1(x)
        x = self.conv4_2(x)
        x = self.activation4_2(x)
        x = self.conv4_3(x)
        x = self.activation4_3(x)
        x = self.pool4(x)
        
        x = self.conv5_1(x)
        x = self.activation5_1(x)       
        x = self.conv5_2(x)
        x = self.activation5_2(x)
        x = self.conv5_3(x)
        x = self.activation5_3(x)
        
        #=======
        #Recurrent Module
        #=======
        
        
        x = self.pool_r(x)
        
        x = x.view(x.shape[0], -1)
        x = torch.unsqueeze(x, 1)
        
        x, h = self.gru(x, h.detach())
        
        x = x[:,-1]
        x = torch.reshape(x, (x.shape[0], 512, 2, 1))
        
        x = self.upsample_r(x)
        
        #=======
        #Decoder
        #=======
        
        x = self.conv6_1(x)
        x = self.activation6_1(x)
        x = self.conv6_2(x)
        x = self.activation6_2(x)
        x = self.conv6_3(x)
        x = self.activation6_3(x)
        x = self.upsample6(x)
        
        x = self.conv7_1(x)
        x = self.activation7_1(x)
        x = self.conv7_2(x)
        x = self.activation7_2(x)
        x = self.conv7_3(x)
        x = self.activation7_3(x)
        x = self.upsample7(x)
        
        x = self.conv8_1(x)
        x = self.activation8_1(x)
        x = self.conv8_2(x)
        x = self.activation8_2(x)
        x = self.conv8_3(x)
        x = self.activation8_3(x)
        x = self.upsample8(x)
        
        x = self.conv9_1(x)
        x = self.activation9_1(x)
        x = self.upsample9(x)
        x = self.conv9_2(x)
        x = self.activation9_2(x)
        x = self.upsample9(x)
        
        x = self.conv10_1(x)
        x = self.activation10_1(x)
        x = self.conv10_2(x)
        x = self.activation10_2(x)
        
        x = self.output(x)
        x = self.activation_output(x)
        
        return x, h
    
    
    def name(self):
        return "Generator"
    
    
    def load_SalGan_weights(self, pretrained_weights='gen_modelWeights0090.npz'):
        
        self.pretrained_weights = np.load(open(pretrained_weights, 'rb'), allow_pickle=True)
        
        self.layers = [module for module in self.modules() if type(module) != nn.Sequential]
        self.conv_layers = [conv for conv in self.layers if type(conv) == nn.modules.conv.Conv2d or type(conv) == nn.modules.conv.ConvTranspose2d]

        array_idx = 0
        for layer in self.conv_layers:

            if type(layer) == nn.modules.conv.ConvTranspose2d:
                #the dim order of weight shape is messed up for ConvTranspose, so do transpose before loading

                self.reshaped_array = self.pretrained_weights['arr_' + str(array_idx)].transpose(1, 0, 2, 3)

                if layer.weight.shape == self.reshaped_array.shape:
                    layer.weight = nn.Parameter(torch.from_numpy(self.reshaped_array).float())
                else:
                    print("One layer was initialized with Xavier due to shape mismatch")
                    torch.nn.init.xavier_uniform_(layer.weight)           
                
                array_idx += 1  #add 1 to idx to fetch the next element from the file

                self.reshaped_array = self.pretrained_weights['arr_' + str(array_idx)]
                
                if layer.bias.shape == self.reshaped_array.shape:
                    layer.bias = nn.Parameter(torch.from_numpy(self.reshaped_array.copy()).float())
                else:
                    layer.bias.data.fill_(0.01)

            else: 
                layer.weight = nn.Parameter(torch.from_numpy(self.pretrained_weights['arr_' + str(array_idx)]).float())

                array_idx += 1  #add 1 to idx to fetch the next element from the file

                layer.bias = nn.Parameter(torch.from_numpy(self.pretrained_weights['arr_' + str(array_idx)]).float())

            array_idx += 1 #add 1 to idx to fetch the next element from the file
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x
    
    
class TemporalDiscriminator(nn.Module):
    '''
    ideal input shape is n, 1, 114, 114
    
    '''
    def __init__(self):
        super(TemporalDiscriminator, self).__init__()
        
        self.DenseNet = torchvision.models.densenet121(pretrained=True)
        self.DenseNet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.DenseNet.features.pool0 = Identity()
        self.DenseNet.fc = nn.Linear(in_features=1000, out_features=1000, bias=True)
        
        self.gru = nn.GRU(input_size=1000, hidden_size=512, num_layers=3, bias=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(512, 1, bias=True)
        
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    @autocast()
    def forward(self, x, h):
        
        x = self.DenseNet(x)
        x = torch.unsqueeze(x, 1)
        x, h = self.gru(x, h.detach())
        x = x[:,-1]
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x, h
    
    
class StyleGanDiscriminator(nn.Module):
    def __init__(self):
        super(StyleGanDiscriminator, self).__init__()
        
        with open('network-snapshot-000720.pkl', 'rb') as f:
            self.StyleGan = pickle.load(f)['D'].cuda() 
            self.sigmoid = nn.Sigmoid()
        
    #@autocast()    
    def forward(self, x):
        
        x = self.StyleGan(x, None)
        x = self.sigmoid(x)
        
        return x
        
    
    def name(self):
        return "StyleGanDiscriminator"
    
    
class StyleGanGenerator(nn.Module):
    def __init__(self):
        super(StyleGanGenerator, self).__init__()
        
        self.DenseNet = torchvision.models.densenet121(pretrained=True)
        self.DenseNet.fc = nn.Linear(in_features=1000, out_features=1000, bias=True)
            
        self.gru = nn.GRU(input_size=1000, hidden_size=512, num_layers=3, bias=True, batch_first=True, dropout=0)
        
        with open('network-snapshot-000720.pkl', 'rb') as f:
            self.StyleGan = pickle.load(f)['G_ema'].cuda() 
        
    #@autocast()    
    def forward(self, x, h):

        ## ENCODER
        x = self.DenseNet(x)
        
        ## GRU
        x = torch.unsqueeze(x, 1)
        x, h = self.gru(x, h.detach())
        x = torch.squeeze(x, 1)
        
        ## DECODER
        x = self.StyleGan(x, None)
        
        return x, h
        
    
    def name(self):
        return "StyleGanGenerator"
