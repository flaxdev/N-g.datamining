import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from src.models import HeightmapModel
from src.geometry import Grid, CoordinateList
from scipy.interpolate import griddata

def plotSurface(ax, cmap, x, y, z, g):

  ax.plot_surface(
    x,
    y,
    z,
    facecolors=cmap.to_rgba(g),
    cmap=cm.jet,
    linewidth=0,
    alpha=0.67,
    antialiased=False
  )

def plotGravityField(receiver, z):

  """
  def plotGravityField
  Plots a grid or spots of the gravity field
  """

  fig = plt.figure()
  ax = fig.gca(projection="3d")

  minn, maxx = z.min(), z.max()
  norm = matplotlib.colors.Normalize(minn, maxx)
  cmap = plt.cm.ScalarMappable(norm=norm, cmap="jet")

  # When plotting a coordinate list
  if isinstance(receiver, CoordinateList):

    # Attempt to interpolate the points to a grid
    try:

      plt.title("Gravity anomaly solution on an interpolated grid")

      interpGrid = Grid(100, 100, heightmap=receiver._heightmap)
      interpGridX, interpGridY = np.meshgrid(interpGrid.xx, interpGrid.yy)

      interpolatedGrid = griddata((receiver.xx, receiver.yy), z, (interpGridX, interpGridY), method="linear")

      plotSurface(ax, cmap, interpGridX, interpGridY, interpGrid.zz, interpolatedGrid)

    except Exception:

      plt.title("Gravity anomaly solution at points")

      # Cannot interpolate the grid. Maybe a cross section?
      if isinstance(receiver.heightmap, HeightmapModel):
        plt.contour(receiver.heightmap.E, receiver.heightmap.N, receiver.heightmap.H, levels=50)

    # Always plot the actual station locations
    ax.scatter(
      receiver.xx,
      receiver.yy,
      receiver.zz,
      color=cmap.to_rgba(z),
      s=100
    )

  # When plotting a grid we can span a surface
  elif isinstance(receiver, Grid):

    plt.title("Gravity anomaly solution on a regular grid")

    xx, yy = np.meshgrid(receiver.xx, receiver.yy)

    # Plot x, y, z as a heightmap with the colors based on the gravity data
    plotSurface(ax, cmap, xx, yy, receiver.zz, z)

  plt.colorbar(cmap)
  plt.show()
