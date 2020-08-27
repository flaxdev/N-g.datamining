import numpy as np
import pandas as pd

from src.geometry import Grid

def saveTimeseries(filepath, iterator, receiver, timeseries, names): 

  # Information header
  header = "\t".join(names)

  np.savetxt(filepath, timeseries, delimiter="\t", header=header)

def loadMetadata(filepath):

  df = pd.read_csv(filepath, sep="\t", header=0)

  x = df["x"].to_numpy()
  y = df["y"].to_numpy()

  return x, y, df["name"]
