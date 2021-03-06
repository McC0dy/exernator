import numpy as np
from numpy.fft import fft
from numpy.fft import fftfreq
from utils import normalize

def crosscorrelate(data):
  corr = np.correlate(data, data, 'same')

  return normalize(corr)

# discards the waveLengths that are shorter than half of the video
def waveLength(signal):
  signalFFT = fft(signal)
  signalFFTAbs = abs(signalFFT) # find the lengths of the amplitudes
  sortedArgs   = np.argsort(-signalFFTAbs) # sort them ascending
  
  # find the wave length with the largest amplitued that is <= len(signal)/2
  freqs = fftfreq(len(signal))
  for arg in sortedArgs:
    if abs(1./freqs[arg]) <= len(signal)/2. :
      return int(abs(1/freqs[arg]))
    # else:
    #   print("waveLength discarded: " + str(1./freqs[arg]))