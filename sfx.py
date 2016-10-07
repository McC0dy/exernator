##
# Special effects
#
##

import cv2
import numpy as np
from utils import saveFrames
from math import floor

def crossfade(cap1, cap2, N):
  """ Takes 2 VideoCapture's cap1 and cap2 and fades cap1 to cap2 in N frames """
  frames = []
  gamma  = 0
  for weight in np.linspace(0, 1, N):

    _, frame1 = cap1.read()
    _, frame2 = cap2.read()
    #print cap1.get(cv2.CAP_PROP_POS_FRAMES)

    # insert any effect here! : 
    frame = cv2.addWeighted(frame1, 1-weight, frame2, weight, gamma)
    frames.append(frame)

    cv2.imwrite("intermediateFrames/alpha=" + str(weight) + ".png", frame)

  return frames

# use crossfade to loop a repetition
def loopRepetitionFade(filename, start, repLength, nLoops, fadeFrames, transitionFunc, videoWriter):
  frames = []
  cap1 = cv2.VideoCapture(filename)
  cap2 = cv2.VideoCapture(filename)

  # Skip to the start frame
  cap1.set(cv2.CAP_PROP_POS_FRAMES, start)

  # First time we show the first frames with NO fading
  if nLoops > 0:
    for _ in range(fadeFrames):
      _, frame = cap1.read()
      #print cap1.get(cv2.CAP_PROP_POS_FRAMES)
      frames.append(frame)
    saveFrames(videoWriter, frames)

    # construct the repetitions with a transition in between :
    frames = []
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start+fadeFrames)
    # show a repetition
    for _ in range(repLength - 2*fadeFrames):
      _, frame = cap1.read()
      #print cap1.get(cv2.CAP_PROP_POS_FRAMES)
      frames.append(frame)

    # for debugging; save the frame before warping
    # cv2.imwrite("intermediateFrames/start2.jpg", frames[-1])
    # cv2.imwrite("intermediateFrames/start1.jpg", frames[-10])

    # In the end of a repetition fade from the end to the start
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = frames + transitionFunc(cap1, cap2, fadeFrames) # 'transitionFunc' is a func-arg

    # save the repetitions we just created
    for _ in range(nLoops-1): # in the last repetition there's no fade
      saveFrames(videoWriter, frames)

    # show one last repetition with no fade at the end
    frames = []
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start+fadeFrames)
    for _ in range(fadeFrames, repLength):
      _, frame = cap1.read()
      #print cap1.get(cv2.CAP_PROP_POS_FRAMES)
      frames.append(frame)
    saveFrames(videoWriter, frames)

  # we're making an infinite video that consists of only 1 rep with fading
  else:

    frames = []
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start+fadeFrames)
    # show a repetition
    for _ in range(repLength - 2*fadeFrames):
      _, frame = cap1.read()
      #print cap1.get(cv2.CAP_PROP_POS_FRAMES)
      frames.append(frame)

    saveFrames(videoWriter, frames)
    # for debugging; save the frame before warping
    # cv2.imwrite("intermediateFrames/start2.jpg", frames[-1])
    # cv2.imwrite("intermediateFrames/start1.jpg", frames[-10])

    # In the end of a repetition fade from the end to the start
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = transitionFunc(cap1, cap2, fadeFrames) # 'transitionFunc' is a func-arg
    saveFrames(videoWriter, frames)

  return #frames # not needed


# plays the following: (start to end + end to start)* + start to end
def loopRepetitionReverse(filename, start, repLength, nLoops, videoWriter):
  cap = cv2.VideoCapture(filename)
  #framesReversed = [None] * repLength
  frames         = [None] * repLength
  cap.set(cv2.CAP_PROP_POS_FRAMES, start)

  # get the frames that we will loop
  for i in range(repLength):
    _, frame = cap.read()
    frames[i] = frame

  # play the frames forwards and backwards
  for _ in range(int(nLoops/2*2 +1)): # nLoops must be a multiple of odd - this rounds up
    saveFrames(videoWriter, frames)
    frames.reverse() # Too heavy? - naaaa

  return #frames + frames.reverse() # not needed

def cutOutRepetition(filename, start, repLength, nLoops, fadeFrames, transitionFunc, videoWriter):
  frames = []
  cap1 = cv2.VideoCapture(filename)
  cap2 = cv2.VideoCapture(filename)

  # Skip to the start frame
  cap1.set(cv2.CAP_PROP_POS_FRAMES, start)

  # Skip to the end frame
  cap2.set(cv2.CAP_PROP_POS_FRAMES, start+repLength)
  frames = transitionFunc(cap1, cap2, fadeFrames) # 'transitionFunc' is a func-arg

  # save the fade we just created
  saveFrames(videoWriter, frames)
