from src.geometry import Position

class PropertyIterator():

  """
  Class PropertyIterator
  Linearly iterates over properties from start to end in n steps
  """

  def __init__(self, iterators):

    self.iterators = iterators
    self.steps = sum(map(lambda x: x.steps, iterators))
    self._currentIterator = self.iterators.pop(0)

  def __iter__(self):
    return self

  def __next__(self):

    if not self._currentIterator.active:

      if len(self.iterators) == 0:
        raise StopIteration

      self._currentIterator = self.iterators.pop(0)

    return self._currentIterator.__next__()

class LinearPropertyIterator():

  """
  class LinearPositionIterator
  Returns an iterator that interpolates between start / end to simulate a moving anomaly
  """

  def __init__(self, steps, props):

    self.steps = steps
    self.properties = dict(props)

    # Keep track of current step
    self._step = -1

  def reversed(self):
    return LinearPropertyIterator(self.steps, {x: tuple(reversed(y)) for (x, y) in self.properties.items()})

  @property
  def fields(self):
    return [*self.properties]

  def __getattr__(self, name):

    if name not in self.properties:
      raise KeyError("The property '%s' is not available in the iterator." % name)

    # Unpack the start and end of the linear interpolation
    try:
      (start, end) = self.properties.get(name)
   # Constant value just return the number
    except TypeError:
      return self.properties.get(name)

    frac = self._step / (self.steps - 1)

    # Explicitly handle Position classes
    if isinstance(start, Position) and isinstance(end, Position):
      x = start.x + frac * (end.x - start.x)
      y = start.y + frac * (end.y - start.y)
      z = start.z + frac * (end.z - start.z)
      return Position(x, y, z)

    return start + frac * (end - start)

  def __iter__(self):
    return self

  def __next__(self):

    if not self.active:
      raise StopIteration

    self._step += 1

    return self
   
  @property
  def active(self):
    return self._step < (self.steps - 1)

  def __str__(self):
    return "\n".join(["<%s: %s>" % (x, self.__getattr__(x).__str__()) for x in self.properties.keys()])
