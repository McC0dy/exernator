import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from utils import normalize, lowPass
from warp import findInitialFlow, findFlow


def dist(start, end):
  startX = start[0,0]
  startY = start[0,1]
  endX = end[0,0]
  endY = end[0,1]
  dist = math.pow(startX-endX, 2)
  dist+= math.pow(startY-endY, 2)
  dist = math.sqrt(dist)
  return dist

def frameDistances(filename):
  """ For all frames in a video find the average movements of features
        from one frame to the next. Return a list of the average movements
  """
  cap  = cv2.VideoCapture(filename)

  _, frame = cap.read()
  frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  distances = []

  # for the entire video do:
  while True:
    frame_old  = frame
    ret, frame = cap.read()
   
    if (ret == True):
      frame      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
      # find features to track 
      #cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)
      corners = cv2.goodFeaturesToTrack(frame, 100, 0.1, 10)
      nextPts, status, err = cv2.calcOpticalFlowPyrLK(frame_old, frame, corners, None)

      # movement in total
      distTotal = 0
      # for each feature calc the euclidean distance
      for (start, end) in zip(corners, nextPts):
        distTotal += dist(start, end)

      # normalize it since we might have found fewer features:
      if len(nextPts) == 0:
        distTotal = 0 # the special case where no features where found
      else:
        distTotal /= len(nextPts) 
      distances.append(distTotal)

    else:
      distances = normalize(distances)
      np.savetxt(filename + ".data", distances)
      return distances


# THIS IS NOT AN ESTIMATE: THIS IS THE REAL DEAL
def frameDistances2(filename):
  """ For all frames in a video find the average movements
        from one frame to the next. Return a list of the average movements
  """
  cap  = cv2.VideoCapture(filename)

  _, frame = cap.read()
  frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  SCALEX = 0.1     # factor used when downscaling
  SCALEY = SCALEX   # factor used when downscaling

  frame = cv2.resize(frame, dsize=(0,0), fx=SCALEX, fy=SCALEY)

  distances = []
  firstRun = True

  # for the entire video do:
  while True:
    frame_old  = frame
    ret, frame = cap.read()
   
    if (ret == True):
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
      frame = cv2.resize(frame, dsize=(0,0), fx=SCALEX, fy=SCALEY)

      if firstRun:
        flow = findInitialFlow(frame_old, frame)
        firstRun = False
      else:
        flow = findFlow(frame_old, frame, flow)
      flowSize = np.sum(flow)
      #print (flowSize)

      distances.append(flowSize)

    else:
      distances = normalize(distances)
      #np.savetxt(filename + ".data", distances)
      return distances



def plotAverageDistances(filename):
  cap = cv2.VideoCapture(filename)
  distances = frameDistances(filename)
  fps   = cap.get(cv2.CAP_PROP_FPS)
  frameMs = int(1000/fps)

  x = range(0, len(distances)*frameMs, frameMs)
  plt.plot(x, distances, 'b') # now draw the movement

  # plot the manually found repetition times
  #plt.plot(repTimes, np.zeros(len(repTimes)), 'rv')

  plt.title("Optical flow as a measure of repetitions")
  plt.xlabel("Time [ms]")
  plt.ylabel("Average feature displacement [pixel]")

  plt.show()

  return

def findSmallestOpticalFlow(distances, length):

  #distances = lowPass(distances, k=5) # we smoothen in main
  opticalFlowMin = abs(distances[0] + distances[length]) # initiate the min-val

  for lengthMultiple in [length]:#range(length, len(distances), length):
    for n in range(0, len(distances)-lengthMultiple, 1):
      opticalFlow = abs(distances[n] + distances[n+lengthMultiple])
      # save the new minimum!
      if opticalFlow < opticalFlowMin:
        opticalFlowMin  = opticalFlow
        nBest           = n
        lengthBest      = lengthMultiple
        #print ("nBest, lengthBest = " + str(nBest) + ", " + str(lengthBest))

  return nBest, lengthBest