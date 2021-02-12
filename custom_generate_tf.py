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
import dnnlib.tflib as tflib

def generate_images(arrs, network_pkl, truncation_psi=1.0,noise_mode='const', outdir='out', save=True, seed=1):
    
    """
    Generates images from an array of latent vectors
    Saves to outdir if save==True
    Returns an array of nchw images 
    """
    
    tflib.init_tf()
    
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, G = pickle.load(fp)
        
    os.makedirs(outdir, exist_ok=True)
    imgs=[]

    # Render images for dlatents initialized from random seeds.
    G_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False,
        'truncation_psi': truncation_psi
    }

    noise_vars = [var for name, var in G.components.synthesis.vars.items() if name.startswith('noise')]
    label = np.zeros([1] + G.input_shapes[1][1:])
    
    for idx, w in enumerate(arrs):
        rnd = np.random.RandomState(seed)
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = G.run(w, label, **G_kwargs) # [minibatch, height, width, channel]
        if save:
            PIL.Image.fromarray(images[0], 'RGB').save(f'{outdir}/{idx:04d}.png')
        
        print(f'Generated {idx}/{len(arrs)-1}')
        imgs.append(images[0])
    return imgs
