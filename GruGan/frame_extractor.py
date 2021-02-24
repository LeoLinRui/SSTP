from cv2 import VideoCapture, imwrite, resize, INTER_AREA
import os
import matplotlib.pyplot as plt

def extract_frames(video: str, outdir: str, dim=(256,256)):

  '''
  Extracts frames from from video, resizes to dim and saves to outdir

  params:
  video: path to video
  outdir: output directory
  dim: dimension tuple of (height, width) 

  '''
  
  print(f"Extracting frames from {video} at resolution {dim[0]}x{dim[1]}")

  if outdir is not None and not os.path.exists(outdir):
    os.mkdir(outdir)
    
  vidcap = VideoCapture(video)
  success = True
  idx = 0

  success,img = vidcap.read()
  while success:
    
    img = resize(img, dim, interpolation = INTER_AREA) # cv2.INTER_AREA resize 

    if (outdir is not None) and (img is not None):
      path = os.path.join(outdir,f"frame{idx:05d}.jpg")
      imwrite(path, img)
      print(f'Saved {path}')
    
    success,img = vidcap.read()
    idx += 1

  print(f"Extracted {idx} frames from {video} at resolution {dim[0]}x{dim[1]}")
