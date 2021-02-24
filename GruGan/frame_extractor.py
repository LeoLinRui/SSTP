from cv2 import VideoCapture
from cv2 import imwrite
import os
import matplotlib.pyplot as plt

def extract_frames(video: str, outdir=None):
  
  if outdir is not None and not os.path.exists(outdir):
    os.mkdir(outdir)
    
  vidcap = VideoCapture(video)
  success = True
  idx = 0
  while success:
    success,img = vidcap.read()

    if (outdir is not None) and (img is not None):
      path = os.path.join(outdir,f"frame{idx:05d}.jpg")
      imwrite(path, img)
      print(f'Saved {path}')
      
    idx += 1

  print("Done!")
