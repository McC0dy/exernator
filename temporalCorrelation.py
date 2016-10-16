import numpy as np
import cv2

def sim(i1, i2):
  return np.sum((i1-np.mean(i1)) * (i2-np.mean(i2)))

def similarities(fn):

  cap1 = cv2.VideoCapture(fn)

  frames = loadFrames(fn)

  len = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

  sims = np.empty((len,len))

  for n1 in range(len):
    print(n1)

    for n2 in range(len):
      s = sim(frames[n1], frames[n2])
      sims[n1,n2] = s

  return sims

def loadFrames(fn):
  SCALEX = 0.2      # factor used when downscaling
  SCALEY = SCALEX   # factor used when downscaling

  frames = []
  cap = cv2.VideoCapture(fn)

  len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  for i in range(len):
    _, frame   = cap.read()
    frameSmall = cv2.resize(frame, dsize=(0,0), fx=SCALEX, fy=SCALEY)
    frames.append(frameSmall)

  return frames

def plot2d(fn):
  sims = similarities(fn)

  plt.title("Temporal Correlation")
  plt.xlabel("Frame I1")
  plt.ylabel("Frame I2")

  plt.imshow(sims, cmap='gray')
  plt.show()