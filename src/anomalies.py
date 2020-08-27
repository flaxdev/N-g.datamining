import numpy as np

from functools import partial

from scipy.special import legendre
from src.geometry import Grid

class Anomaly():

  """
  class Anomaly
  Parent class of all anomalies
  """

  G = 6.67408E-11 # Gravitationl Constant
  POISSON = 0.25 # Poisson ratio
  FAG = 0.309E-5 # Free air gradient

  def __init__(self, position, density):

    self.position = position
    self.density = density

  def setPosition(position):
    self.position = position

  def simulate(self, x):

    """
    def simulate
    Wrapper simulation function: scale the internal simulation result to mgal
    """

    return 1E5 * self._simulate(x)


class ThinSheetAnomaly(Anomaly):

  def __init__(self, position, density):

    Anomaly.__init__(self, position, density, strike)

    self.strike = 80

  def _simulate(self, grid):
    # Rotate the coordinates in line with the strike
    pass

class OkuboAnomaly(Anomaly):

  """
  Class OkuboAnomaly

  Container for an Okubo anoomaly (gravity & deformation)

  Implemented after Okubo, 1992 (10.1029/92JB00178) with the help of the
  MatLab implementation of Okubo92 by François Beauducel.

  See the attached copyright notice: https://nl.mathworks.com/matlabcentral/fileexchange/37158-okubo-gravity-change-due-to-shear-and-tensile-faults
  """

  def __init__(self, position, density, dip, strike, rake, length, width, slip, normal, fill=None):
   
    Anomaly.__init__(self, position, density)

    # Source parameters
    self.dip = np.radians(dip)
    self.strike = np.radians(strike)
    self.rake = np.radians(rake)
    self.length = length
    self.width = width
    self.slip = slip
    self.normal = normal
    self.fill = fill

  def _simulate(self, receiver):

    """
    def _simulate
    Internal simulation of the Okubo displacement anomaly
    """

    # Relative receiver positions
    dx = receiver.xx - self.position.x
    dy = receiver.yy - self.position.y
    dz = receiver.zz - self.position.z

    # Grid
    if isinstance(receiver, Grid):
      dx, dy = np.meshgrid(dx, dy)

    # Create the displacement vector <U1, U2, U3>
    # Strike Slip
    U1 = np.cos(self.rake) * self.slip
    # Dip Slip
    U2 = np.sin(self.rake) * self.slip
    # Normal to plane (tensile component)
    U3 = self.normal 

    # Convert fault coordinates relative to centroid
    # Move center of fault down to the base (Okubo convention)
    depth = dz + np.sin(self.dip) * 0.5 * self.width
    ec = dx + np.cos(self.strike) * np.cos(self.dip) * 0.5 * self.width
    nc = dy - np.sin(self.strike) * np.cos(self.dip) * 0.5 * self.width

    # Coordinates
    x = np.cos(self.strike) * nc + np.sin(self.strike) * ec + 0.5 * self.length
    y = np.sin(self.strike) * nc - np.cos(self.strike) * ec + np.cos(self.dip) * self.width 
    p = y * np.cos(self.dip) + depth * np.sin(self.dip);
    q = y * np.sin(self.dip) - depth * np.cos(self.dip);

    # Eqs. (58), (59), and (60) in order
    slipStrike = U1 * self._chinnery(self._strikeHeight, x, p, q)
    slipDip = U2 * self._chinnery(self._dipHeight, x, p, q)
    tensile = U3 * self._chinnery(self.tensileHeight, x, p, q)

    # Calculate dilatation offset (height) following Eq. (57)
    dh = (1. / (2 * np.pi)) * (slipStrike + slipDip + tensile)

    # Eqs. (52), (53), and (54) in order
    slipStrike = U1 * self._chinnery(self._strikeGravity, x, p, q)
    slipDip = U2 * self._chinnery(self._dipGravity, x, p, q)
    tensile = U3 * self._chinnery(self.tensileGravity, x, p, q)
    
    # Material that fills the cavity
    if self.fill is None:
      cavityFill = 0
    else:
      cavityFill = (self.fill - self.density) * self.G * U3 * self._chinnery(self._cavityGravity, x, p, q)

    # Free air gradient
    gradient = self.FAG * dh

    # Combine all effects
    return self.density * self.G * (slipStrike + slipDip + tensile) + cavityFill - gradient 
         

  def _R(self, xi, eta, q):

    """
    def _R
    Helper function to calculate the euclidean distance
    """

    return np.sqrt(xi ** 2 + eta ** 2 + q ** 2)

  def _yBar(self, eta, q):

    """
    def _yBar
    Helper function to calculate y with a bar on top from Eq. (56)
    """

    return eta * np.cos(self.dip) + q * np.sin(self.dip)

  def _dBar(self, eta, q):

    """
    def _yBar
    Helper function to calculate d with a bar on top  from Eq. (36)
    """

    return eta * np.sin(self.dip) - q * np.cos(self.dip)

  def _cavityGravity(self, xi, eta, q):

    """
    def _cavityGravity
    Returns the gravity effect of a cavity fill
    """

    R = self._R(xi, eta, q)

    return 2 * self._I2(xi, eta, q, R) * np.cos(self.dip) - \
           np.sin(self.dip) * np.log(R + xi)

  def _strikeGravity(self, xi, eta, q):

    """
    def _strikeGravity
    Returns the strike slip effect on gravity
    """

    R = self._R(xi, eta, q)

    return -(q * np.sin(self.dip) / R) + \
           ((q ** 2 * np.cos(self.dip)) / (R * (R + eta)))

  def _dipGravity(self, xi, eta, q):

    """
    def _dipGravity
    Returns the dip slip effect on gravity
    """

    R = self._R(xi, eta, q)
    db = self._dBar(eta, q)

    return 2 * self._I2(xi, eta, q, R) * np.sin(self.dip) - \
           ((q * db) / (R * (R + xi)))

  def tensileGravity(self, xi, eta, q):

    """
    def tensileHeight
    Returns the tensile opening term for gravity
    """

    R = self._R(xi, eta, q)
    yb = self._yBar(eta, q)

    return 2 * self._I2(xi, eta, q, R) * np.cos(self.dip) + \
           (q * yb) / (R * (R + xi)) + \
           (q * xi * np.cos(self.dip)) / (R * (R + eta))


  def _strikeHeight(self, xi, eta, q):

    """
    def _strikeHeight
    Returns the Strike Slip height term
    """

    R = self._R(xi, eta, q)
    db = self._dBar(eta, q)

    # Eq. 58
    return -(db * q) / (R * (R + eta)) - \
           (q * np.sin(self.dip)) / (R + eta) - \
           self._I4(db, eta, q, R) * np.sin(self.dip)

  def tensileHeight(self, xi, eta, q):

    """
    def tensileHeight
    Returns the tensile opening term for height
    """

    R = self._R(xi, eta, q)
    db = self._dBar(eta, q)
    yb = self._yBar(eta, q)

    return (yb * q) / (R * (R + xi)) + \
           np.cos(self.dip) * ((xi * q) / (R * (R + eta)) - \
           np.arctan((xi * eta) / (q * R))) - \
           self._I5(db, xi, eta, q, R) * np.sin(self.dip) ** 2
    

  def _dipHeight(self, xi, eta, q):

    """
    def _dipHeight
    Returns the dip slip term for height
    """

    R = self._R(xi, eta, q)
    db = self._dBar(eta, q)

    return -((db * q) / (R * (R + xi))) - \
           np.sin(self.dip) * np.arctan((xi * eta) / (q * R)) + \
           self._I5(db, xi, eta, q, R) * np.sin(self.dip) * np.cos(self.dip)

  def _I1(self, xi, eta, q, R):

    """
    def _I1
    Returns the I1 term
    """

    # Eq. (31)
    return np.arctan((-q * np.cos(self.dip) + (1 + np.sin(self.dip)) * (R + eta)) / (xi * np.cos(self.dip)))

  def _I2(self, xi, eta, q, R):

    """
    def _I2
    Returns the I2 term
    """

    return np.arctan((R + xi + eta) / q)

  def _I4(self, db, eta, q, R):

    """
    def _I4
    Returns the I4 term
    """

    # Eq. (61) or Eq. (63) when the dip is (close to) 90
    if np.isclose(np.cos(self.dip), 0):
      return -(1 - 2 * self.POISSON) * (q / (R + db))
    else:
      return (1 - 2 * self.POISSON) * (np.log(R + db) - \
             np.sin(self.dip) * np.log(R + eta)) * (1 / np.cos(self.dip))

  def _I5(self, db, xi, eta, q, R):

    """
    def _I5
    Returns the I5 term
    """

    # Eq. (62) or Eq. (64) when the dip is (close to) 90
    if np.isclose(np.cos(self.dip), 0):
      return -(1 - 2 * self.POISSON) * ((xi * np.sin(self.dip)) / (R + db))
    else:
      return 2 * (1 - 2 * self.POISSON) * self._I1(xi, eta, q, R) * (1 / np.cos(self.dip))


  def _chinnery(self, f, x, p, q):

    """
    def _chinnery
    Use Chinnery's notation for the given substitution:

    f(xi, eta)|| = f(x, p) - f(x, p - W) - f(x - L, p) + f(x - L, p - W)

    """

    # Closed form notation: see Chinnery, 1961
    return f(x, p, q) \
           - f(x, p - self.width, q) \
           - f(x - self.length, p, q) \
           + f(x - self.length, p - self.width, q)


class PrismAnomaly(Anomaly):

  """
  class PrismAnomaly
  Container for a prism shaped density anomaly
  """

  def __init__(self, position, density, area, top, bottom):

    Anomaly.__init__(self, position, density)

    self.area = area
    self.top = top
    self.bottom = bottom

  def _simulate(self, receiver):

    """
    def _simulate
    Forward simulation of the prism anomaly
    """

    prefix = self.G * self.density * self.area

    dx = receiver.xx - self.position.x
    dy = receiver.yy - self.position.y
    h1 = receiver.zz - self.top
    h2 = receiver.zz - self.bottom

    if isinstance(receiver, Grid):
      dx, dy = np.meshgrid(dx, dy)

    # Eq. 4.3b
    return prefix * ((1 / (dx ** 2 + dy ** 2 + h1) ** 0.5) - \
                    (1 / (dx ** 2 + dy ** 2 + h2) ** 0.5))


class CylindricalAnomaly(Anomaly):

  """
  class CylindricalAnomaly
  Forward model for gravity of a cylindrical anomaly at a particular position, density, radius
  """

  # Legendre Polynomials
  _P2 = legendre(2)
  _P4 = legendre(4)
  _P6 = legendre(6)
  _P8 = legendre(8)
  _P10 = legendre(10)

  def __init__(self, position, density, radius):

    Anomaly.__init__(self, position, density)

    self.radius = radius
    self.density = density

  def _legendrePrime(self, cosTheta, r):

    """
    def _legendrePrime
    Coefficients following eq. 4.15 for r > R
    """

    fact = self.radius / r

    # First five coefficients given by 4.15
    coefficients = np.array([
      +(1/2) * fact,
      -(1/8) * fact ** 3,
      +(1/16) * fact ** 5,
      -(5/128) * fact ** 7,
      +(7/256) * fact ** 9
    ])

    # P0 - P8: P0 is a matrix of ones.
    values = np.array([
      np.ones(cosTheta.shape),
      self._P2(cosTheta),
      self._P4(cosTheta),
      self._P6(cosTheta),
      self._P8(cosTheta)
    ])

    return np.sum(values * coefficients, axis=0)

  def _legendre(self, cosTheta, r):
 
    """
    def _legendrePrime
    Coefficients following eq. 4.14 for r <= R
    """

    fact = r / self.radius

    coefficients = np.array([
      -fact,
      +(1/2) * fact ** 2,
      -(1/8) * fact ** 4,
      +(1/16) * fact ** 6,
      -(5/128) * fact ** 8,
      +(7/256) * fact ** 10
    ])

    # Apply the matrix
    values = np.array([
      cosTheta,
      self._P2(cosTheta),
      self._P4(cosTheta),
      self._P6(cosTheta),
      self._P8(cosTheta),
      self._P10(cosTheta)
    ])

    return np.sum(values * coefficients, axis=0)

  def _simulate(self, receiver):

    """
    def CylindricalAnomaly._simulate
    Simulates a cylindrical anomly
    """
 
    # Constant at front
    prefix = 2 * np.pi * self.G * self.density * self.radius

    # Subtract in a vectorized way
    dx = receiver.xx - self.position.x
    dy = receiver.yy - self.position.y
    dz = receiver.zz - self.position.z

    if isinstance(receiver, Grid): 
      dx, dy = np.meshgrid(dx, dy)

    # Determine the position to the source of the cylinder
    x = (dx ** 2 + dy ** 2) ** 0.5

    # Defined as angle & radius
    theta = np.cos(np.arctan2(x, dz))
    r = (x ** 2 + dz ** 2) ** 0.5

    # Two solutions for two domains (see p. 30, 31): only valid sometimes
    # Set invalid domain to 0 and sum the result to construct the full model
    one = 1 + self._legendre(theta, r)
    one[r >= self.radius] = 0
    two = self._legendrePrime(theta, r)
    two[r < self.radius] = 0

    # Merge the two solutions
    return prefix * (one + two)


class ThinRodAnomaly(Anomaly):

  def __init__(self, position, density, area, length, angle):

    Anomaly.__init__(self, position, density)

    self.area = area
    self.length = length
    self.angle = np.radians(angle)

  def _simulate(self, receiver):

    # Short hand
    angle = self.angle
    L = self.length

    dx = receiver.xx - self.position.x
    dy = receiver.yy - self.position.y
    dz = receiver.zz - self.position.z

    if isinstance(receiver, Grid):
      dx, dy = np.meshgrid(dx, dy)

    # Should be horizontal distance from O
    x = (dx ** 2 + dy ** 2) ** 0.5

    cot = (1. / np.tan(angle))
    csc = (1. / np.sin(angle))

    # Eq. 4.2...
    termA = (x + dz * cot) / ((dz ** 2 * csc ** 2 + (2 * x * dz * cot) + x ** 2) ** 0.5)
    termB = (x + dz * cot + (L * np.cos(angle))) / (((L + dz * csc) ** 2 + x ** 2 + (2 * x * (L * np.cos(angle) + dz * cot))) ** 0.5)

    return ((self.G * self.density * self.area) / (x * np.sin(angle))) * (termA - termB)
 
class MogiAnomaly(Anomaly):

  def __init__(self, position, density, radius, dvolume):

    Anomaly.__init__(self, position, density)

    self.radius = radius
    self.dvolume = dvolume

  def _simulate(self, receiver):

    """
    def _simulate
    Forward simulation of the spherical gravity anomaly (eq. 4.1)
    """

    # Mogi point source deformation
    mogiPrefix = self.dvolume * ((1 - self.POISSON) / np.pi)

    # 4.1
    prefix = (4 / 3) * np.pi * self.G * self.density * self.radius ** 3

    # These are matrices for speed
    dx = receiver.xx - self.position.x
    dy = receiver.yy - self.position.y
    dz = receiver.zz - self.position.z

    if isinstance(receiver, Grid):
      dx, dy = np.meshgrid(dx, dy)

    dr = dx ** 2 + dy ** 2 + dz ** 2

    # Graivty & vertical displacement
    return (prefix + mogiPrefix * self.FAG) * (dz / dr ** (3 / 2))


class SphericalAnomaly(Anomaly):

  """
  class SphericalAnomaly
  Simple model for a spherical anomaly

  TODO Should add Mogi deformation here 
  """

  def __init__(self, position, density, radius):

    Anomaly.__init__(self, position, density)

    self.radius = radius

  def _simulate(self, receiver):

    """
    def _simulate
    Forward simulation of the spherical gravity anomaly (eq. 4.1)
    """

    # 4.1
    prefix = (4 / 3) * np.pi * self.G * self.density * self.radius ** 3

    # These are matrices for speed
    dx = receiver.xx - self.position.x
    dy = receiver.yy - self.position.y
    dz = receiver.zz - self.position.z

    if isinstance(receiver, Grid):
      dx, dy = np.meshgrid(dx, dy)

    dr = dx ** 2 + dy ** 2 + dz ** 2

    return prefix * dz / dr ** (3 / 2)
