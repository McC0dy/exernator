#!/usr/bin/python3

import cv2
import numpy as np
from opticalFlowMin import frameDistances, findSmallestOpticalFlow, frameDistances2
from crosscorrelation import waveLength, crosscorrelate
from sfx import loopRepetitionFade, loopRepetitionReverse, crossfade, cutOutRepetition
from imageDiffs import bestStart, frameDifferences2
from utils import getFrames, saveVideo, saveFrames, showVideo, lowPass, normalize
from warp import warpedFrames, findInitialFlow, findFlow
import matplotlib.pyplot as plt
from fitToSine import fitSine
import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable


DEBUG = False # TODO: show plots of distances, cross-correlation and fourier transform

def loopVideo(filename, nLoops, fadeFrames, allowReverse):
  gamma       = 0.25 # weight of self-similarity compared to optical flow
  cap         = cv2.VideoCapture(filename)
  videoWriter = saveVideo(filename)

  distances   = frameDistances2(filename)
  differences = frameDifferences2(filename)[:-1]

  distances   = normalize(distances)
  differences = normalize(differences)

  # smoothing performed over an odd number 'k' equal or 1 larger than fadeFrames:
  k = fadeFrames/2 *2 +1
  if k<19:
    k=19

  distances   = lowPass(distances,   k)
  differences = lowPass(differences, k)

  signal = (1-gamma)*distances + gamma*differences
  corr   = crosscorrelate(signal)


#Find the most prominent wavelength
  waveL = waveLength(corr)


  [A, Freq, Phase], succes = fitSine(corr, Freq=1./waveL)
  fitfunc = lambda p, x: p[0] * np.cos(2*np.pi*p[1]*x + p[2])
  
  waveL = int(1./Freq)
  # print(waveL)

  fps   = cap.get(cv2.CAP_PROP_FPS)
  print("length of repetition : " + str(waveL/fps))

  # attempt to find the true curve
  bestStartFrame, bestStartLength = findSmallestOpticalFlow(distances, waveL)
  trueCurve = signal[bestStartFrame : bestStartFrame+bestStartLength]
  repetitionDetection = np.correlate(trueCurve, signal, 'same')


  if allowReverse:
    bestStartFrame, bestStartLength = findSmallestOpticalFlow(distances, waveL)
    print("bestStartFrame  = " + str(bestStartFrame))
    print("bestStartLength = " + str(bestStartLength))

  else:
    # Find the repetition to loop by searching for 2 frames that are multiple of waveL apart
    #   that are the most similar
    bestStartFrame, bestStartLength = bestStart(filename, waveL, fadeFrames)
    print("bestStartFrame  = " + str(bestStartFrame))
    print("bestStartLength = " + str(bestStartLength))


  # get frames up to bestStartFrame, but only if we're making a finite video
  if nLoops != 0:
    frames = getFrames(filename, 0, bestStartFrame-fadeFrames/2)
    saveFrames(videoWriter, frames)
    del frames

  if nLoops < 0:
    cutOutRepetition(filename, bestStartFrame-fadeFrames/2, bestStartLength, nLoops, fadeFrames, warpedFrames, videoWriter)

  else:
    # get the frames in the center of the original movie
    if allowReverse:
      frames = loopRepetitionReverse(filename, bestStartFrame, bestStartLength, nLoops, videoWriter)
      #saveFrames(videoWriter, frames)
      del frames
    else:
      frames = loopRepetitionFade(filename, bestStartFrame-fadeFrames/2, bestStartLength, nLoops, fadeFrames, warpedFrames, videoWriter)
      #frames = loopRepetitionFade(filename, bestStartFrame-fadeFrames/2, bestStartLength, nLoops, fadeFrames, crossfade, videoWriter)
      # saveFrames(videoWriter, frames) # loopRepetitionFade has saved them already
      del frames
  

  # get frames of the end of the video
  if nLoops != 0: 
    frames = getFrames(filename, bestStartFrame+bestStartLength-fadeFrames/2)
    #frames = getFrames(filename, bestStartFrame+bestStartLength+fadeFrames/2)
    saveFrames(videoWriter, frames)
    del frames

  videoWriter.release()


def saveLoop(filename, nLoops=3, fadeFrames = 8, allowReverse = False):

  if allowReverse == True:
    fadeFrames=0 # if reverse playback is allowed, then fading is not applied - independent of what the user said

  # if nLoops < 0:
  #   print ("error: nLoops < 0 ")
  #   exit()
  if fadeFrames < 0:
    print ("error: fadeFrames < 0 ")
    exit()
  if allowReverse != False and allowReverse != True:
    print ("error: allowReverse must be either: True | False")
    exit()

  frames = loopVideo(filename, nLoops, fadeFrames, allowReverse)
