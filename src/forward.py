from src.sources import MogiPointSource

def singleAscendingSource(grid, t, x=500000, y=4180000, dm=1E10, fromDepth=-4000, toDepth=-1000):

  """
  def singleAscendingSource
  Single point source ascending from t=0 to t=1

  Do not know if an ascending mogi point source model makes physical sense..

  """

  # Ascend t from -4000 to -1000 in t steps (t goes from 0 => 1)
  z = (t * (toDepth - fromDepth)) + fromDepth

  # Create the ascending source: position and mass change
  source = MogiPointSource(
    x, y, z,
    dm
  ) 

  return grid.solve(source)
