import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp2d

class HeightmapModel():

  def __init__(self, filepath):

    matlabObject = loadmat(filepath)
    
    self.E = matlabObject["E1"]
    self.N = matlabObject["N1"]
    self.H = matlabObject["H1"]
    
    # Interpolate in 2D
    self.model = interp2d(
      self.E[0,:],
      self.N[:,0],
      self.H.T.ravel()
    )

  def interpolate(self, x, y):
    return self.model(x, y)

class PlaneModel():

  def __init__(self):
    pass

  def interpolate(self, x, y):
    return np.zeros((x.size, y.size))
