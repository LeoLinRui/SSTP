import cv2.VideoCapture
import cv2.imwrite
import os
import matplotlib.pyplot as plt

def extract_frames(video: str, outdir=None):
  vidcap = cv2.VideoCapture(video)
  imgs = []
  success = True

  while success:
    success,img = vidcap.read()
    imgs.append(img)
  
  imgs.pop(-1)

  if outdir is not None:
    os.mkdir(outdir)
    num_imgs = len(imgs)

    for idx, img in enumerate(imgs):
      path = os.path.join(outdir,f"frame{idx:05d}.jpg")
      cv2.imwrite(path, img)
      print(f'Saved {path}...Progress: {idx}/{num_imgs}')

  print("Done!")
  return imgs
