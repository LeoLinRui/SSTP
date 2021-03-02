import glob

import PIL.Image
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc, imread

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

#TODO: This shit sometimes works
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

def numpy2video(arr: np.ndarray, fps=30, mp4_path="np_out.mp4"):
  assert mp4_path.endswith('.mp4')
  
  frameSize= (arr[0].shape[0], arr[0].shape[1])
  out = VideoWriter(mp4_path,VideoWriter_fourcc(*'MP4V'), fps, frameSize)

  for img in arr:
    img = img.astype("uint8")
    out.write(img)

  out.release()
