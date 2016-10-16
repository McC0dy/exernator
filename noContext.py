from temporalCorrelation import loadFrames
import matplotlib.pyplot as plt
import numpy as np
import cv2

fn = "trump.mp4"

#sims = similarities(fn)
#np.savetxt("sims.txt", sims)
#sims = np.loadtxt("sims.txt")

def findMatches():
  simsLen = sims.shape[0] # it's a square matrix

  # linear weighting scheme:
  weights  = np.array([[np.power(abs(x-y),0.5) for x in range(simsLen)] for y in range(simsLen)])
  simsNorm = sims / (weights+0.0000001) # apply weights but avoid division by zero

  # all the diagonals and neighbors have the smallest errors - what to do?!
  for i in range(simsLen):
    simsNorm[i,i] = 9999999

  maxs = [np.argmax(row) for row in simsNorm]
  mins = [np.argmin(row) for row in simsNorm]

  matches = []
  for i in range(simsLen):
    if i == maxs[maxs[i]]: # they are each others best matches
      matches.append((i,maxs[i]))
  return matches

def similarities(fn):

  cap1 = cv2.VideoCapture(fn)

  frames = loadFrames(fn)

  len = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

  sims = np.empty((len,len))

  for n1 in range(len):
    print(n1)

    for n2 in range(len):
      #s = sim(frames[n1], frames[n2])
      s = np.sum(np.power(frames[n1] - frames[n2], 2))
      sims[n1,n2] = s

  return sims