import glob

import PIL.Image
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize, INTER_CUBIC

import numpy as np

#----------------------------------------------------------------------------
def gif_from_folder(imgs_path: str, gif_path='gif/gif.gif', fps=30):
    assert imgs_path.endswith('/*.png')
    assert gif_path.endswith('.gif')
    
    print("Generating gif from images in " + imgs_path)
    
    
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(imgs_path))]
    
    duration = len(imgs)/fps
    
    img.save(fp=gif_path, format='GIF', append_images=imgs,
             save_all=True, duration=duration, loop=0)
    
    print("Done")

#----------------------------------------------------------------------------

def mp4_from_folder(imgs_path: str, mp4_path='video.mp4', fps=30):
    assert imgs_path.endswith('/*.png')
    assert mp4_path.endswith('.mp4')
    
    first = imread(next(x for x in glob.glob(imgs_path)))
    frameSize= (first.shape[0], first.shape[1])
    
    #out = VideoWriter(mp4_path,VideoWriter_fourcc(*'h264'), fps, frameSize)
    out = VideoWriter(mp4_path,VideoWriter_fourcc(*'MP4V'), fps, frameSize)
    
    for filename in sorted(glob.glob(imgs_path)):
        img = imread(filename)
        out.write(img)
        print(filename)

    out.release()
    
#----------------------------------------------------------------------------

def numpy2video(
    arr: np.ndarray, # array of cv2 images
    mp4_path="np_out.mp4", # path to output mp4 video
    fps=30, # framerate
    scale=True, # whether to scale from 0-1 to 0-255
    target_dim=(1920,1080), # tuple of (width, height) for the target video size
    is_color=True # whether the input images are color
    ):
  
  """
  Creates video from aray of cv2-format images

  """
  
  assert mp4_path.endswith('.mp4')
  
  frameSize= (target_dim[0], target_dim[1])
  out = VideoWriter(
      mp4_path,VideoWriter_fourcc(*'MP4V'), fps, frameSize, is_color)
    
  if scale:
    for img in arr:
      img = np.multiply(img, 255).astype("uint8")
      img = resize(img, target_dim, interpolation=INTER_CUBIC)
      out.write(img)
  else:
    for img in arr:
      img = img.astype("uint8")
      img = resize(img, target_dim, interpolation=INTER_CUBIC)
      out.write(img)

  out.release()
