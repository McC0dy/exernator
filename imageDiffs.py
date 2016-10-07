import numpy as np
import cv2
from utils import normalize, lowPass
import matplotlib.pyplot as plt
from warp import findFlow, findInitialFlow
from opticalFlowMin import frameDistances2
from crosscorrelation import waveLength, crosscorrelate
from scipy.signal import medfilt
from fitToSine import fitSine

filename  = '/home/oliver/study/thesis/videos/lowerBack/lowerBack0.mp4'
repTimes  = [9500, 17000, 24000, 32000] # for 'lowerBack0.mp4'

# ?? Returns the differences of every frame and SOME other one defined below
def frameDifferences(filename):

  cap1 = cv2.VideoCapture(filename)
  cap2 = cv2.VideoCapture(filename)

  # skip to the center of the video
  frame_count = cap2.get(cv2.CAP_PROP_FRAME_COUNT)
  cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_count/2)

  _, frameOriginal = cap2.read()
  #frameOriginal    = cv2.cvtColor(frameOriginal, cv2.COLOR_BGR2GRAY)

  ds = []
  while(True):
    ret, frame = cap1.read()
    #frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if (ret):
      d = diff(frameOriginal, frame)
      #print d
      ds.append(d)
    else:
      return ds

# diffs = frameDifferences2("../videos/lowerBack/lowerBack0.mp4")
# finds the correlation between a frame in the middle and all frames in the video
# plots the result
def frameDifferences2(filename, comparisonPos=-1):

  SCALEX = 0.1     # factor used when downscaling
  SCALEY = SCALEX   # factor used when downscaling

  cap1 = cv2.VideoCapture(filename)
  cap2 = cv2.VideoCapture(filename)

  # skip to the frame of comparison
  frame_count = cap2.get(cv2.CAP_PROP_FRAME_COUNT)
  if comparisonPos < 0: # skip to the center of the video
    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_count/2)
  elif comparisonPos <= frame_count: 
    cap2.set(cv2.CAP_PROP_POS_FRAMES, comparisonPos)
  else:
    print ("error: trying to compare frames with a frame that doesn't exist")
    exit()

  # read the center frame and downscale it
  _, frameOriginal = cap2.read()
  i1 =   cv2.resize(frameOriginal, dsize=(0,0), fx=SCALEX, fy=SCALEY)
  #frameOriginal    = cv2.cvtColor(frameOriginal, cv2.COLOR_BGR2GRAY)

  # avg pixel brightness used in the correlation bellow
  #avg1 = np.mean(i1.flatten())
  avg1 = np.mean(i1)

  ds = []
  while(True):
    ret, frame = cap1.read() # get the current frame

    if (ret):
      # downscale it:
      #frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      i2 = cv2.resize(frame, dsize=(0,0), fx=SCALEX, fy=SCALEY)

      #avg2  = np.mean(i2.flatten())
      avg2 = np.mean(i2)
      #d     = 0
      # for all pixels in the 2 frames do:
      # for pixel1,pixel2 in zip(i1.flatten(), i2.flatten()): 
      #   d += (pixel1 - avg1) * (pixel2 - avg2)
      d = np.sum( (i1-avg1) * (i2-avg2) )
      ds.append(d)
    # if the last frame has been read, then plot the correlation
    else:
      # plt.plot(ds)
      # plt.title("Self-similarity of 'lowerBack0' measured by normalized correlation")
      # plt.xlabel("Frame")
      # plt.ylabel("Correlation")
      # #plt.show()
      # plt.savefig("../report/figures/normalizedCorrelation.png")
      return ds


# return the normalized correlation of 2 images
def diff(i1, i2):
  # SCALEX = 0.05     # factor used when downscaling
  # SCALEY = SCALEX   # factor used when downscaling

  # i1 = cv2.resize(image1, dsize=(0,0), fx=SCALEX, fy=SCALEY)
  # i2 = cv2.resize(image2, dsize=(0,0), fx=SCALEX, fy=SCALEY)

  return np.sum((i1-np.mean(i1)) * (i2-np.mean(i2)))


# Looks through an entire video and finds the diff of
#   frame I(t) and I(t+waveLength) returns the smallest
#   frame number with the smallest diff, since it might
#   be a good choice for start frame.
def goodStart(filename, waveL):
  cap1  = cv2.VideoCapture(filename)
  cap2  = cv2.VideoCapture(filename)
  diffs = []
  flows = []
  firstRun = True

  #SCALEX = 0.1     # factor used when downscaling
  SCALEX = 0.05
  SCALEY = SCALEX   # factor used when downscaling

  cap2.set(cv2.CAP_PROP_POS_FRAMES, waveL) # cap2 jumps waveL ahead!

  # for the entire video do:
  while True:
    _, frame1 = cap1.read()
    frame1    = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    ret2, frame2 = cap2.read()
   
    if (ret2 == True):
      frame2    = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

      i1 = cv2.resize(frame1, dsize=(0,0), fx=SCALEX, fy=SCALEY)
      i2 = cv2.resize(frame2, dsize=(0,0), fx=SCALEX, fy=SCALEY)

      d = diff(i1, i2)
      diffs.append(d)

      if firstRun:
        flow = findInitialFlow(i1, i2)
        firstRun = False
      else:
        flow = findFlow(i1, i2, flow)

      flowAbs = np.absolute(flow)
      flows.append(np.sum(flowAbs)) # save the size of the flow

    else:
      diffs = np.array(diffs)
      flows = np.array(flows)
      k = 19 # large number used for smoothing
      # flows = lowPass(flows, k) 
      # diffs = lowPass(diffs, k) 
      diffs = medfilt(diffs, k)
      flows = medfilt(flows, k)

      #flows = normalize(flows) # if normalized, it can't be compared across different waveLength's
      #diffs = normalize(diffs)      # if normalized, it can't be compared across different waveLength's

      flow[flow == 0.] = 1. # avoid division with 0 later

      # plt.plot(flows + diffs)
      # plt.title("flows + diffs")
      # plt.show()4

      # plt.plot(diffs/flows)
      # plt.title("diffs / flows")
      # plt.show()

      return np.array(diffs/flows)

# Find the 2 frames that are multiples of wavelength apart that are the most similar
def bestStart(filename, waveL, fadeFrames):
  cap = cv2.VideoCapture(filename)
  videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  bestStartV = -99999999 # small number
  goodStartV = 0 # just so it's declared
  goodStartN = 0 # just so it's declared

  #print("wavelengths used for finding bestStart:")
  for waveLength in range(waveL, videoLength, waveL):
    print("currently investigating waveLength: " + str(waveLength))
    goodStarts  = goodStart(filename, waveLength)

    # plt.title(str(waveLength))
    # plt.plot(goodStarts)
    # plt.show()

    goodStartNs = np.argsort(-goodStarts) # sort them ascending

    for i in goodStartNs: # skip the frames that are too close to the start and end (we can't warp there!)
      if (i > fadeFrames) and (i+waveLength < videoLength -fadeFrames):
        goodStartN = i
        goodStartV = goodStarts[i] # value: highest correlation
        break

    if goodStartV > bestStartV: # the best start has the highest correlation
      bestStartV = goodStartV
      bestStartN = goodStartN
      bestStartLength = waveLength
  print ("bestStartV = " + str(bestStartV))
  return bestStartN, bestStartLength # frame to begin from AND the video length to show


# returns the repetition length found 
def repLengthVariation(filename):
  cap         = cv2.VideoCapture(filename)
  videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  diffs       = []
  repLengths  = []

  distances   = frameDistances2(filename)
  distances   = normalize(distances)
  
  k = 21 # large k
  distances   = lowPass(distances, k)

  for i in range(videoLength):
    print (str(i) + " / " + str(videoLength))

    diff      = frameDifferences2(filename, comparisonPos=i)
    diff      = normalize(diff)
    diff      = lowPass(diff, k)
    signal    = diff

    corr      = crosscorrelate(signal)
    repLength = waveLength(corr)

    [A, Freq, Phase], succes = fitSine(corr, Freq=1./repLength)
    repLength = int(1./Freq)
    
    repLengths.append(repLength)
    print (repLength)

  return repLengths
