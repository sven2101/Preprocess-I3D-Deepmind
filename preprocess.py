import os
import sys
import cv2
import numpy as np
import time

DATA_DIR = 'data/vids/cricket'
SAVE_DIR = 'out'

_EXT = ['.avi', '.mp4']
_IMAGE_SIZE = 224
_CLASS_NAMES = 'data/label_map.txt'


def get_video_length(video_path):
  _, ext = os.path.splitext(video_path)
  if not ext in _EXT:
    raise ValueError('Extension "%s" not supported' % ext)
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened(): 
    raise ValueError("Could not open the file.\n{}".format(video_path))
  if cv2.__version__ >= '3.0.0':
    CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
  else:
    CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
  length = int(cap.get(CAP_PROP_FRAME_COUNT))
  cap.release()
  return length

def compute_rgb(video_path):
    """Compute RGB"""
    rgb = []
    vidcap = cv2.VideoCapture(video_path)
    success,frame = vidcap.read()
    while success:
        frame = cv2.resize(frame, (342,256)) 
        frame = (frame/255.)*2 - 1
        frame = frame[16:240, 59:283]    
        rgb.append(frame)        
        success,frame = vidcap.read()
    vidcap.release()
    rgb = rgb[:-1]
    rgb = np.asarray([np.array(rgb)])
    print('save rgb with shape ',rgb.shape)
    np.save(SAVE_DIR+'/rgb.npy', rgb)
    return rgb
        

def compute_TVL1(video_path):
  """Compute the TV-L1 optical flow."""
  flow = []
  TVL1 = cv2.DualTVL1OpticalFlow_create()
  vidcap = cv2.VideoCapture(video_path)
  success,frame1 = vidcap.read()
  bins = np.linspace(-20, 20, num=256)
  prev = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
  vid_len = get_video_length(video_path)
  for _ in range(0,vid_len-1):
      success, frame2 = vidcap.read()
      curr = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY) 
      curr_flow = TVL1.calc(prev, curr, None)
      assert(curr_flow.dtype == np.float32)
      
      #Truncate large motions
      curr_flow[curr_flow >= 20] = 20
      curr_flow[curr_flow <= -20] = -20
     
      #digitize and scale to [-1;1]
      curr_flow = np.digitize(curr_flow, bins)
      curr_flow = (curr_flow/255.)*2 - 1
    
      #cropping the center
      curr_flow = curr_flow[8:232, 48:272]  
      flow.append(curr_flow)
      prev = curr
  vidcap.release()
  flow = np.asarray([np.array(flow)])
  print('Save flow with shape ', flow.shape)
  np.save(SAVE_DIR+'/flow.npy', flow)
  return flow

def main():
  start_time = time.time()
  print('Extract Flow...')
  compute_TVL1(DATA_DIR+'/v_CricketShot_g04_c01.avi')
  print('Compute flow in sec: ', time.time() - start_time)
  start_time = time.time()
  print('Extract RGB...')
  compute_rgb(DATA_DIR+'/v_CricketShot_g04_c01.avi')
  print('Compute rgb in sec: ', time.time() - start_time)
  
if __name__ == '__main__':
  main()
