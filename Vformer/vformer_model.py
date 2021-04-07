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

BATCH_SIZE = 24


class CombinedEmbedding(nn.Module):
    def __init__(self, embedding_dim=512, vocab_size=16384, reformer_layer=12, positional_encoding=True, fmap_encoding=True):
        super(CombinedEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.reformer_layer = reformer_layer
        self.positional_encoding = positional_encoding
        self.fmap_encoding = fmap_encoding

        self.token_embedder = nn.Linear(in_features=self.vocab_size, out_features=self.embedding_dim, bias=False)
        self.positional_embedder = nn.Embedding(num_embeddings=256, embedding_dim=self.embedding_dim)
        self.fmap_embedder = nn.Embedding(num_embeddings=256, embedding_dim=self.embedding_dim)

        self.device = torch.device('cpu')


    def forward(self, x, position_idx=None):
        self.position_idx = position_idx

        assert torch.max(x) == 1 and torch.min(x) == 0, "input has to be one-hot encoded. Got max {}, min {}".format(torch.max(x), torch.min(x))
        x.to(self.device)

        # if input is just one image, shape == (b, 1, 256, vocab_size)
        if position_idx is not None :
            if len(x.shape) is 3:
                pass
            elif len(x.shape) is 4 and x.shape[1] is 1:
                x = x.squeeze(1)
            else:
                raise ValueError('input shape {} is incorrect for position_idx not None'.format(x.shape))
                    
            out = torch.zeros(x.shape[0], x.shape[1], x.shape[2], self.embedding_dim, device=self.device)
            for i, s in enumerate(x): #traverse sequences in a batch
                for j, m in enumerate(s): #traverse feature maps in a frame
                    out[i][j] = self.token_embedder(m)
                    if self.fmap_encoding:
                            out[i][j] += self.fmap_embedder(torch.tensor(position_idx, device=self.device).long())

        # if input is a full sequence, shape == (b, 256, 256, vocab_size)
        else:
            if self.positional_encoding:
                assert len(x.shape) == 4, 'no position_idx given for 3 dimension input'

            out = torch.zeros(x.shape[0], x.shape[1], x.shape[2], self.embedding_dim, device=self.device)
            for i, s in enumerate(x): #traverse sequences in a batch
                for j, f in enumerate(s): #traverse frames in a sequence
                    for k, m in enumerate(f): #traverse feature maps in a frame
                        out[i][j][k] = self.token_embedder(m) 
                        if self.positional_encoding:
                            out[i][j][k] += self.positional_embedder(torch.tensor(j, device=self.device).long()) 
                        if self.fmap_encoding:
                            out[i][j][k] += self.fmap_embedder(torch.tensor(k, device=self.device).long())

        return out
    
    
    def get_embedding_matrix(self):
        return list(self.token_embedder.parameters())[0]


    def to(self, device):
        assert isinstance(device, torch.device)
        self.device = device

        self.token_embedder = nn.Linear(in_features=self.vocab_size, out_features=self.embedding_dim, bias=False).to(self.device)
        self.positional_embedder = nn.Embedding(num_embeddings=256, embedding_dim=self.embedding_dim).to(self.device)
        self.fmap_embedder = nn.Embedding(num_embeddings=256, embedding_dim=self.embedding_dim, ).to(self.device)
        
        return self


    def cuda(self):
        self.to(torch.device('cuda'))
        return self
    
    
class Conv(nn.Module):
    def __init__(self, dic_size=20000, output_len=256):
            super(Conv, self).__init__()
            
            self.dic_size = dic_size
            self.output_len = output_len
            assert math.sqrt(self.output_len).is_integer(), "sqrt of output_len has to be an integer"

            self.maxpool1 = nn.MaxPool2d(kernel_size=[3, 3], stride=2)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
            self.softmax = nn.Softmax(dim=2)
            self.relu = nn.ReLU()

            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[7, 7], stride=1, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[7, 7], stride=2, padding=3, bias=False)
            self.bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.conv4 = nn.Conv2d(in_channels=128, out_channels=self.dic_size, kernel_size=[3, 3], stride=1, padding=1)


    #@autocast()
    def forward(self, x):
        def forward_single(x):
            b = x.shape[0]

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = F.adaptive_max_pool2d(x, output_size=int(math.sqrt(self.output_len)*4))

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)

            x = self.maxpool1(x)
            x = self.conv4(x)

            x = F.adaptive_avg_pool2d(x, output_size=int(math.sqrt(self.output_len)))

            ## input shape: (b, c, h, w)
            x = x.view(b, x.shape[1], x.shape[2] * x.shape[3]) ## shape: (b, c, h*w)
            x = F.gumbel_softmax(x, tau=0.2, hard=True, eps=1e-10, dim=-1) ## shape: (b, c, h*w)
            x = x.transpose(1, 2) ## shape: (b, h*w, c)

            return x
        
        if len(x.shape) == 4: ## input shape: (b, c, h, w)
            return forward_single(x) ## output shape: (b, h*w, c)
        
        elif len(x.shape) == 5: ## input shape: (b, f, c, h, w)
            out = torch.empty(x.shape[0], x.shape[1], self.output_len, self.dic_size, device=x.device)
            for f in range(x.shape[1]):
                out[:,f,:,:] = forward_single(x[:,f,:,:,:])
            return out ## output shape: (b, f, h*w, c)
        
        else:
            raise ValueError('input shape {} is incorrect'.format(x.shape))
                
            
class DeConv(nn.Module):
    def __init__(self, input_dim=1024):
        super(DeConv, self).__init__()
        self.input_dim = input_dim

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.deconv1 = nn.ConvTranspose2d(in_channels=self.input_dim, out_channels=512, kernel_size=[4, 4], stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=[4, 4], stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=[4, 4], stride=2, padding=1)

    
    #@autocast()
    def forward(self, x):
        b = x.shape[0]

        x = x.view(b, 16, 16, self.input_dim)
        x = x.transpose(1, 3).transpose(2, 3)
        x = self.deconv1(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.sigmoid(x)

        return x
    
    
class Reformer(nn.Module):
    def __init__(self, vocab_size):
        super(Reformer, self).__init__()
        self.vocab_size = vocab_size
        
        ## load the model from huggingface, slice off the embedding, input, and output layers
        self.model = transformers.ReformerModelWithLMHead.from_pretrained("google/reformer-enwik8")
        self.model = list(self.model.children())
        self.model = list(self.model[0].children())
        self.model = self.model[1]
        
        self.output = nn.Linear(2048, self.vocab_size, bias=True)

        
    #@autocast()
    def forward(self, x):
        
        x = self.model(x)
        x = self.output(x)
        x = F.softmax(x)

        return x