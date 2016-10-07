import numpy as np
import cv2

def normalize(l):
  l = (l - np.mean(l)) / (np.std(l) * len(l))
  return l

def lowPass(s, k=5):
  if (k<1) or (k % 2 == 0):
    print ("error: k<1 or (k % 2 != 0)")
    exit()
  if k > len(s):
    k = len(s) + (len(s) % 2) -1

  kHalf = min(int(k/2), len(s)) # the windows shouldn't be larger than the data
  sSmooth = np.zeros_like(s)

  # go through the signal and smooth it out
  for i in range(1,kHalf):
    sSmooth[i] = np.mean(s[:i])
  for i in range(kHalf, len(s)-kHalf):
    sSmooth[i] = np.mean(s[i-kHalf : i+kHalf])
  for i in range(len(s)-kHalf, len(s)):
    sSmooth[i] = np.mean(s[i:])

  return sSmooth

def getFrames(filename, startFrame, nFrames=99999):
  nFrames = int(nFrames)
  frames = []
  cap  = cv2.VideoCapture(filename)

  # Skip to startFrame
  cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)

  # get the frames
  for _ in range(nFrames):
    ret, frame = cap.read()
    if ret == False: # if we reach the end of the video, then return
      return frames
    frames.append(frame)

  return frames

def saveVideo(filename):
  cap = cv2.VideoCapture(filename)

  # get the info from the 'old' video
  filenameNew = 'out.avi'
  fourcc      = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
  fps         = cap.get(cv2.CAP_PROP_FPS)
  width       = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height      = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  size        = (int(width), int(height))

  # get a writer-object for the new video
  wri = cv2.VideoWriter(filenameNew, fourcc, fps, size, isColor=False)
  if not wri.isOpened():
    print ("Error: Unable to write to file: '" + filenameNew + "' - possibly mismatching encoding and filename extension!")
    return False

  ret = wri.open(filenameNew, fourcc, fps, size)

  return wri

def saveFrames(videoWriter, frames):
  for frame in frames:
    videoWriter.write(frame)


def showVideo(frames):
  for frame in frames:
    cv2.imshow("frame", frame)
    cv2.waitKey(30)
    del frame