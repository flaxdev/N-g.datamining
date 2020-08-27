class MogiPointSource():

  """
  Class MogiPointSource
  Container for a Mogi Point Source (spherical deformation)
  """

  # Poison ratio
  v = 0.25

  def __init__(self, x, y, z, dm, rho=3300):

    # Position and change of mass
    self.x = x
    self.y = y
    self.z = z
    self.dm = dm

    # Density
    self.rho = rho
