from scipy import optimize
import numpy as np
from numpy import pi, r_
import matplotlib.pyplot as plt

# takes data and initial sine-parameters used when guessing
def fitSine(data, Amplitude=.001, Freq=1/180., Phase=.0):
  Tx = np.linspace(0, len(data)-1, len(data))

  # declare fit and error-functions
  fitfunc = lambda p, x: p[0] * np.cos(2*pi*p[1]*x + p[2]) # Target function
  errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function

  p0 = [Amplitude, Freq, Phase] # Initial guess for the parameters
  p1, success = optimize.leastsq(errfunc, p0, args=(Tx, data))

  return p1,success