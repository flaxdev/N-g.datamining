import numpy as np

from src.models import PlaneModel, HeightmapModel

from src.models import PlaneModel, HeightmapModel


class Geometry():
  # Bounding box of the Mt. Etna heightmap
  NORTH = 4196813
  EAST = 519351
  SOUTH = 4156013
  WEST = 480801

  def __init__(self, heightmap):

    if heightmap == "etna":
      self.heightmap = HeightmapModel("./data/etna.mat")
      self._heightmap = "etna"

    elif heightmap == "plane":
      self.heightmap = PlaneModel()
      self._heightmap = "plane"


class CoordinateList(Geometry):

  def __init__(self, x, y, heightmap="etna"):
    Geometry.__init__(self, heightmap)

    self.nx = len(x)
    self.ny = len(y)

    self.xx = np.array(x)
    self.yy = np.array(y)
    self.zz = np.array([self.heightmap.interpolate(x, y) for (x, y) in zip(self.xx, self.yy)]).flatten()

  @property
  def shape(self):
    return self.nx


class Position():
  """
  class Position
  Container for an x, y, z world position
  """

  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z

  def __str__(self):
    return "<x: %.2f, y: %.2f, z: %.2f>" % (self.x, self.y, self.z)


class Grid(Geometry):
  """
  class Grid
  Container for an x, y grid using a meshgrid useful for vector ops.
  """

  def __init__(self, nx, ny, heightmap="etna"):
    Geometry.__init__(self, heightmap)

    self.nx = nx
    self.ny = ny

    self.xx = np.linspace(self.WEST, self.EAST, self.nx)
    self.yy = np.linspace(self.SOUTH, self.NORTH, self.ny)

    # Station height is constrained by the heightmap (for now)
    self.zz = self.heightmap.interpolate(self.xx, self.yy)

  @property
  def shape(self):
    return (self.nx, self.ny)

class Geometry():
  # Bounding box of the Mt. Etna heightmap
  NORTH = 4196813
  EAST = 519351
  SOUTH = 4156013
  WEST = 480801

  def __init__(self, heightmap):

    if heightmap == "etna":
      self.heightmap = HeightmapModel("./data/etna.mat")
      self._heightmap = "etna"

    elif heightmap == "plane":
      self.heightmap = PlaneModel()
      self._heightmap = "plane"


class CoordinateList(Geometry):

  def __init__(self, x, y, heightmap="etna"):
    Geometry.__init__(self, heightmap)

    self.nx = len(x)
    self.ny = len(y)

    self.xx = np.array(x)
    self.yy = np.array(y)
    self.zz = np.array([self.heightmap.interpolate(x, y) for (x, y) in zip(self.xx, self.yy)]).flatten()

  @property
  def shape(self):
    return self.nx


class Position():
  """
  class Position
  Container for an x, y, z world position
  """

  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z

  def __str__(self):
    return "<x: %.2f, y: %.2f, z: %.2f>" % (self.x, self.y, self.z)


class Grid(Geometry):
  """
  class Grid
  Container for an x, y grid using a meshgrid useful for vector ops.
  """

  def __init__(self, nx, ny, heightmap="etna"):
    Geometry.__init__(self, heightmap)

    self.nx = nx
    self.ny = ny

    self.xx = np.linspace(self.WEST, self.EAST, self.nx)
    self.yy = np.linspace(self.SOUTH, self.NORTH, self.ny)

    # Station height is constrained by the heightmap (for now)
    self.zz = self.heightmap.interpolate(self.xx, self.yy)

  @property
  def shape(self):
    return (self.nx, self.ny)

class Grid():

  """
  class Grid
  Creates a (nx, ny) grid on which gravity from a source will be evaluated
  """

  def __init__(self, x, y, z):

    # Create the grid spacings
    self.x = x
    self.y = y
    self.z = z

  def solveSteps(self, length, model):

    """
    def Grid.solveSteps
    Solves the forward problem over a number of time steps
    """

    slices = list()

    # Apply the model from over a unit time interval
    for step in np.linspace(0, 1, length):
      slices.append(model(self, step))

    # Stack the data in the right shape for timeseries
    # Return model consistent with the reach of the code
    return np.vstack(slices)

  def solve(self, source):

    """
    def Grid.solve
    Fast vectorized solver of gravity on a grid
    """

    # Physical parameters
    G = 6.67E-11
    anoise = 10

    # Mogi point source volume change
    mogi = ((source.dm / source.rho) * (1 - source.v) / np.pi)

    # Subtract the source position from the virtual gravimeter positions
    dx = self.x - source.x
    dy = self.y - source.y
    dz = self.z - source.z

    # Vectorized solve for distance to source
    r2 = (dx ** 2 + dy ** 2 + dz ** 2) ** (3/2)

    # Vertical displacement because of volume change
    mz = mogi * (dz / r2)

    # Bouguer slab approximation
    bg = 111.9 * mz

    # And the free air gradient (in microgal)
    fag = 308.6 * mz

    # Add some noise
    noise = anoise * np.random.rand(self.x.size)

    # Without deformation
    return noise + (1E8 * G * (((source.dm) * dz / r2))) 

    # With mogi deformation
    # return noise - fag + (1E8 * G * (((source.dm) * dz / r2) + bg)) 
