"""
Helpers for creating/generating vectors
"""
import numpy as np


def seed2vec(seed, latent_shape):
  return np.random.RandomState(seed).randn(*latent_shape)
  
#----------------------------------------------------------------------------

def generate_transition_latent(arr1, arr2, frames=60):
    
    '''
    Generates all latents vectors between two arrays.
    
    Final array does not include the last vector
    '''
    
    assert arr1.shape == arr2.shape
    
    assert arr1.shape
    
    delta = np.divide(np.subtract(arr2, arr1), frames)
    out = np.empty((frames, arr1.shape[1], arr1.shape[2]), dtype=float)
    
    for frame in range(frames):
        out[frame] = arr1 + np.multiply((frame), delta)
        
    return out
    
#----------------------------------------------------------------------------

def transition_latent_from_key(arr): 
    
    latent_size = arr[0][0].shape
    
    for element in arr:
        assert type(element) is tuple
        assert len(element) == 2
        assert element[0].shape == latent_size
        latent_size == element[0].shape
    
    for i in range(len(arr)):
        
        try:
            transition_frame_set = generate_transition_latent(arr[i][0], arr[i+1][0], frames=arr[i][1])
        except IndexError:
            transition_frame_set = arr[i][0]
        
        try:
            output = np.vstack((transition_frame_set, output))

        except NameError:
            output = transition_frame_set
            
    return output
