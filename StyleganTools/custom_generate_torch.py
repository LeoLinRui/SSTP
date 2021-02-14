import argparse
import os
import pickle
import re
import glob

import numpy as np
import PIL.Image
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc, imread

import dnnlib
import torch
import legacy

def generate_images(arrs, network_pkl, truncation_psi=1.0,noise_mode='const', outdir='out', save=True, seed=1):
    
    """
    Generates images from an array of latent vectors
    Saves to outdir if save==True
    Returns an array of nchw images 
    """
    
    device = torch.device('cuda')
    
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        
    os.makedirs(outdir, exist_ok=True)
    imgs=[]

    label = torch.zeros([1, G.c_dim], device=device)
    
    for idx, w in enumerate(arrs):
        z = torch.from_numpy(w).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        if save:
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
        
        print(f'Generated {idx}/{len(arrs)-1}')
        imgs.append(img)
    return imgs
