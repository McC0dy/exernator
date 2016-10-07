import numpy as np
import cv2
from scipy import interpolate
from scipy import ndimage

def findInitialFlow(img1_gray, img2_gray):
    initialFlow=0
    return cv2.calcOpticalFlowFarneback(
            prev=img1_gray, next=img2_gray, 
            flow=initialFlow, 
            pyr_scale=.5, 
            levels=5, 
            winsize=15,
            iterations=3, 
            poly_n=5, 
            poly_sigma=1.2,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )

def findFlow(img1_gray, img2_gray, initialFlow):
    return cv2.calcOpticalFlowFarneback(
            prev=img1_gray, next=img2_gray, 
            flow=initialFlow, 
            pyr_scale=.5, 
            levels=5,
            winsize=15, 
            iterations=3, 
            poly_n=5, 
            poly_sigma=1.2,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW
            )
    
# warp using optical flow
def testWarp2():
  #alpha  = 0.5
  flow   = 0 # we don't know the flow for the first run
  frames = []
  cap    = cv2.VideoCapture(filename)

  for alpha in np.linspace(0,1,10):
    _, img1   = cap.read()
    _, img2   = cap.read()

    img_warped, flow = warpedFrame(img1, img2, flow, alpha)
    frames.append(img_warped)

  for frame in frames:
    cv2.imshow("img_warped", frame)
    cv2.waitKey(30)

def warpedFrames(cap1, cap2, fadeFrames):
  # go back 1 frame to calculate the initial flow:
  cap1.set(cv2.CAP_PROP_POS_FRAMES, float(cap1.get(cv2.CAP_PROP_POS_FRAMES)-1.0))
  cap2.set(cv2.CAP_PROP_POS_FRAMES, float(cap2.get(cv2.CAP_PROP_POS_FRAMES)-1.0))
  _, img1   = cap1.read()
  _, img2   = cap2.read()
  img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  dimY      = img1_gray.shape[0]
  dimX      = img1_gray.shape[1]
  flow      = findInitialFlow(img1_gray, img2_gray)

  frames = []

  for alpha in np.linspace(0,1,fadeFrames):
    _, img1   = cap1.read()
    _, img2   = cap2.read()

    img_warped, flow = warpedFrame(img1, img2, flow, alpha)
    frames.append(img_warped)

    # save the warped frame for debugging
    cv2.imwrite("intermediateFrames/alpha=" + str(alpha) + ".png", img_warped)

    #display the flow for debugging:
    # cv2.imshow("flowX", flow[:,:,0])
    # cv2.waitKey(100000)
    # cv2.imshow("flowY", flow[:,:,1])
    # cv2.waitKey(100000)

  return frames


def warpedFrame(img1, img2, initialFlow, alpha=0.5):

  #convert frames to grayscale
  img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  dimX = img1_gray.shape[1]
  dimY = img1_gray.shape[0]

  # determine the flow. this takes ~90% of time
  flow = findFlow(img1_gray, img2_gray, initialFlow)

  # with no optical flow, these are the indices in frame2 corresponding to frame1
  #   AND the indices in frame1 corresponding to frame1 - should we not use the same optical flow?
  Xs1, Ys1 = np.meshgrid(np.linspace(0,dimX-1,dimX), np.linspace(0,dimY-1,dimY))
  Xs2, Ys2 = (Xs1, Ys1)

  # divide the flow into X- and Y-components and scale it with alpha
  # consider interpolating colors before round()!
  flowX2 = ((1-alpha)*flow[:,:,1])
  flowY2 = ((1-alpha)*flow[:,:,0])

  flowX1 = (alpha*flow[:,:,1])
  flowY1 = (alpha*flow[:,:,0])

  # add the optical flow to the indices and 
  #   threshold vals so they are all inside the image
  Xs1 = threshold(Xs1-flowY1, dimX)
  Ys1 = threshold(Ys1-flowX1, dimY)
  Xs2 = threshold(Xs2+flowY2, dimX)
  Ys2 = threshold(Ys2+flowX2, dimY)

  # Xs and Ys are used as indices so they must be ints
  Xs1 = Xs1.round().astype(int)
  Ys1 = Ys1.round().astype(int)
  Xs2 = Xs2.round().astype(int)
  Ys2 = Ys2.round().astype(int)

  # the final image is the weighted average of the interpolation of frame1 and frame2
  img_warped = (1.0-alpha)* img1[Ys1,Xs1] + alpha*img2[Ys2, Xs2]

  img_warped = img_warped.round().astype(np.uint8) # not as uint8 but 'original type'

  return (img_warped, flow)

def threshold(coordinates, dim):
  lowValsIndexes = (coordinates < 0)
  coordinates[lowValsIndexes] = 0
  largeValsIndexes = (coordinates >= dim-1)
  coordinates[largeValsIndexes] = dim-1

  return coordinates

# warp using feature-points and splines
def testWarp():
  cap  = cv2.VideoCapture(filename)

  _, img1   = cap.read()
  img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

  cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
  _, img2   = cap.read()
  img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  img_warped = warp(img1, img2)
  cv2.imshow("img_warped", img_warped)
  cv2.waitKey(100000)


def warp(img1, img2):
  img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  prevPts, nextPts = opticalFlowPts(img1_gray, img2_gray)
  drawMatches(img1, img2, prevPts, nextPts)
  opticalFlowGridX, opticalFlowGridY = fillGrid(prevPts, nextPts, img1.shape[1], img2.shape[0])
  img_warped = propagateImg(img1, opticalFlowGridX, opticalFlowGridY, alpha=0.5)

  return img_warped


def opticalFlowPts(img1, img2):
  maxFeaturePoints = 200

  prevPts = cv2.goodFeaturesToTrack(img1, maxFeaturePoints, 0.1, 10)
  nextPts, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, prevPts, np.array([]))

  #flow = drawMatches(img1, img2, prevPts, nextPts)
  return prevPts, nextPts

def fillGrid(prevPts, nextPts, dimX, dimY):
  xs=[]; ys=[]; dxs=[]; dys=[];
  for (pt1, pt2) in zip(prevPts, nextPts):
    y1  = int(pt1[0,0])
    y2  = int(pt2[0,0])
    x1  = int(pt1[0,1])
    x2  = int(pt2[0,1])

    xs.append(x1)
    ys.append(y1)
    dxs.append(x1-x2)
    dys.append(y1-y2)

  # find interpolation function for optical flows:
  opticalFlowFuncX = interpolate.interp2d(xs, ys, dxs, kind='linear') # consider kind='cubic'
  opticalFlowFuncY = interpolate.interp2d(xs, ys, dys, kind='linear') # consider kind='cubic'

  # generate a grid of optical flow based on the interpolation function
  opticalFlowX = opticalFlowFuncX(range(dimX), range(dimY))
  opticalFlowY = opticalFlowFuncY(range(dimX), range(dimY))

  # cv2.imshow("Optical Flow X Grid", gridX)
  # cv2.waitKey(0)
  # cv2.imshow("Optical Flow Y Grid", gridY)
  # cv2.waitKey(0)

  return opticalFlowX, opticalFlowY


def propagateImg(img2, opticalFlowX, opticalFlowY, alpha=0.5):
  img_warped = np.zeros(img2.shape, dtype=np.uint8)

  for x in range(img_warped.shape[0]):
    for y in range(img_warped.shape[1]):
      xnew = int(round(x + alpha*opticalFlowX[x,y]))
      ynew = int(round(y + alpha*opticalFlowY[x,y]))
      if (xnew<0 or xnew>=img2.shape[0] or ynew<0 or ynew>=img2.shape[1]):
        img_warped[x,y] = 0 # consider another strategy
      else:
        img_warped[x,y] = img2[xnew,ynew]

  return img_warped

def drawMatches(img1, img2, prevPts, nextPts):
  imgCombined = np.concatenate((img1, img2), axis=1)
  dimX = img1.shape[0]
  dimY = img1.shape[1]

  for (pt1, pt2) in zip(prevPts, nextPts):
    y1  = int(pt1[0,0])
    y2  = int(pt2[0,0])
    x1  = int(pt1[0,1])
    x2  = int(pt2[0,1])

    pt1 = (y1,x1)
    pt2 = (y2+dimY, x2)
    col = hslToRgb(float(x1)/dimX, 1, 0.8)
    cv2.line(imgCombined, pt1, pt2, col, 1, 8)

  cv2.imshow("Matches", imgCombined)
  cv2.imwrite("matches.png", imgCombined)
  cv2.waitKey(100000)
  return imgCombined


#testWarp2() # RUN THIS TO PROFILE IT!!


## HELPER FUNCTIONS ##

# The following is not written by me. It is borrowed from somewhere, but I've lost the source!
def hslToRgb(h, s, l):
  if l < 0.5:
    q = l * (1 + s) 
  else: 
    q = l + s - l * s
  p = 2 * l - q
  r = hue2rgb(p, q, h + 1.0/3)
  g = hue2rgb(p, q, h)
  b = hue2rgb(p, q, h - 1.0/3)

  return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)));

def hue2rgb(p, q, t):
  if t < 0:
    t += 1
  if t > 1:
    t -= 1
  if t < 1.0/6:
    return p + (q - p) * 6 * t
  if t < 1.0/2:
    return q
  if t < 2.0/3:
    return p + (q - p) * (2/3 - t) * 6
  return p
